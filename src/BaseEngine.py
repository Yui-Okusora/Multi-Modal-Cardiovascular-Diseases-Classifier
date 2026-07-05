import torch
import time
import os
import csv
import math

class BaseExecutionEngine:
    """
    🎯 CENTRAL RUNTIME ORCHESTRATOR: Highly scalable functional optimization loop.
    Tracks deep hardware allocation telemetry, structural step gradients, 
    and outputs system capacity analytics both to console prints and persistent CSV log ledgers.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.telemetry = {}
        self.used_tag = []

    def _compute_grad_norm(self, parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def compute_variance_loss(self, z, target_std=1.0, eps=1e-4):
        B, K, D = z.size()
        if B <= 1: 
            return torch.tensor(0.0, device=z.device)
            
        # Variance calculated across the batch dimension [B], resulting in [K, D]
        std = torch.sqrt(z.var(dim=0) + eps)
        
        # 🎯 THE FIX: Clamp against 0.85 instead of a rigid 1.0
        return torch.mean(torch.clamp(target_std - std, min=0.0))

    def compute_covariance_loss(self, z):
        B, K, D = z.size()
        if B <= 1: return torch.tensor(0.0, device=z.device)
        z_centered = z - z.mean(dim=0, keepdim=True)
        loss = 0.0
        diagonal_mask = torch.eye(D, device=z.device)
        for k in range(K):
            cov = (z_centered[:, k, :].T @ z_centered[:, k, :]) / (B - 1)
            loss += torch.log1p((cov * (1.0 - diagonal_mask)) ** 2).sum() / D
        return loss / K
    
    def compute_cross_slot_orthogonal_loss(self, z):
        """
        🎯 THE TOKEN SHIELD: Penalizes slots for learning identical features.
        Forces the 8 distinct slot matrices to remain mutually orthogonal.
        Expects z shape: [B, K, D]
        """
        B, K, D = z.size()
        if B <= 1: return torch.tensor(0.0, device=z.device)
        
        # 1. Normalize across the feature dimension to calculate pure directional correlation
        z_norm = torch.nn.functional.normalize(z, p=2, dim=-1)
        
        # 2. Compute cross-slot similarity matrices for every patient session in the batch
        # [B, K, D] @ [B, D, K] -> [B, K, K]
        slot_similarity_matrices = torch.bmm(z_norm, z_norm.transpose(1, 2))
        
        # 3. Define an identity matrix anchor representing perfect slot isolation
        identity_anchor = torch.eye(K, device=z.device).unsqueeze(0).expand(B, -1, -1)
        
        # 4. Minimize the off-diagonal similarity elements across slots
        cross_slot_error = (slot_similarity_matrices - identity_anchor) ** 2
        
        return cross_slot_error.sum() / (B * K * K)
    
    def create_warmup_cosine_scheduler(self, optimizer, num_warmup_steps: int, num_total_steps: int, min_lr_ratio: float = 0.0):
        """
        🎯 THE SCHEDULER BARRIER:
        Builds a unified LambdaLR scheduler that executes a linear warmup phase
        before transitioning smoothly into a cosine annealing decay curve.
        """
        def lr_lambda(current_step: int):
            # 1. Linear Warmup Phase
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # 2. Cosine Annealing Decay Phase
            progress = float(current_step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps))
            # Ensure progress doesn't overshoot due to loose step bounds
            progress = min(max(progress, 0.0), 1.0)
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Scale multiplier smoothly to handle the optional floor limit
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _execute_epoch_loop(self, tag, models, optimizer, data_loader, loss_fn_lambda, num_epochs=10, scheduler=None, before_step = None, after_step = None):
        # Initialize structured multi-metric tracking dictionaries natively
        self.used_tag.append(tag)

        self.telemetry[tag] = {
            "epoch": [], "batch": [], "global_step": [],
            "loss": [], "grad_norm": [], "lr": [],
            "vram_gb": [], "samples_per_sec": [],
            "total_duration": 0.0
        }
        
        metrics = self.telemetry[tag]

        trainable_params = [p for m in models for p in m.parameters() if p.requires_grad]
        global_step_idx = 0
        
        print(f"\n🚀 Initiating High-Order Optimization Pass: [{tag.upper()}] | Budget: {num_epochs} Epochs")
        loop_start_time = time.perf_counter()
        
        for epoch in range(num_epochs):
            for m in models: m.train()
            running_telemetry = {}
            epoch_start = time.perf_counter()
            
            for batch_idx, batch in enumerate(data_loader):
                # ⏱️ TIME CAPTURE START: Monitor throughput speed
                batch_start = time.perf_counter()
                
                optimizer.zero_grad()
                
                # Extract runtime sample payloads standardly
                batch_size = batch['feature_ids'].size(0) if 'feature_ids' in batch else self.cfg.batch_size
                
                # Execute the abstract loss evaluation closure
                loss_output = loss_fn_lambda(batch, global_step_idx, len(data_loader) * num_epochs)

                total_loss = 0.0
                component_logs = []

                loss_dict = loss_output if isinstance(loss_output, dict) else {"total": loss_output}

                for k, val in loss_dict.items():
                    short_name = k.replace("loss_", "")
                    weight, raw_loss = val if isinstance(val, (list, tuple)) else (None, val)
                    
                    weighted_loss = (weight * raw_loss) if weight is not None else raw_loss
                    total_loss += weighted_loss
                    
                    if weight is not None:
                        metrics.setdefault(f"{k}_raw", []).append(raw_loss.item())
                        metrics.setdefault(f"{k}_weighted", []).append(weighted_loss.item())
                        component_logs.append(f"{short_name}:{raw_loss.item():.3f}(x{weight})")
                    else:
                        metrics.setdefault(k, []).append(raw_loss.item())
                        if k != "total":  # Avoid redundant printing during fallback runs
                            component_logs.append(f"{short_name}:{raw_loss.item():.3f}")

                total_loss.backward()

                # Process gradients and enforce rigid structural guards
                grad_norm = self._compute_grad_norm(trainable_params)
                torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip_norm)
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()

                if after_step is not None:
                    after_step()
                    
                # ⏱️ TIME CAPTURE END: Calculate compute compute velocity per second
                batch_end = time.perf_counter()
                step_duration = batch_end - batch_start
                samples_per_sec = batch_size / step_duration if step_duration > 0 else 0.0
                
                # Hardware and optimizer metadata profiling
                current_lr = optimizer.param_groups[0]['lr']
                vram_usage_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                
                # Commit data packets to organized local memory slots
                
                metrics["epoch"].append(epoch + 1)
                metrics["batch"].append(batch_idx)
                metrics["global_step"].append(global_step_idx)
                metrics["loss"].append(total_loss.item())
                metrics["grad_norm"].append(grad_norm)
                metrics["lr"].append(current_lr)
                metrics["vram_gb"].append(vram_usage_gb)
                metrics["samples_per_sec"].append(samples_per_sec)
                
                global_step_idx += 1
                
                if batch_idx % self.cfg.log_interval == 0 or batch_idx == len(data_loader) - 1:
                    comp_str = " ｜ ".join(component_logs) if component_logs else "No active sub-components"
                    print(
                        f"⚡ [{tag:<12}] E{epoch+1:02d} B{batch_idx:03d}/{len(data_loader):03d} │ "
                        f"L_tot: {total_loss.item():.4f} │ G: {grad_norm:5.2f} │ "
                        f"LR: {current_lr:.1e} │ {vram_usage_gb:.1f}GB │ {samples_per_sec:.0f}sam/s\n"
                        f"🧬 {comp_str}"
                    )
                          
            print(f"--- [{tag}] Epoch {epoch+1:02d} Complete | Duration: {time.perf_counter() - epoch_start:.2f}s ---\n")
            
        total_duration = time.perf_counter() - loop_start_time
        metrics["total_duration"] = total_duration

    def _generate_and_save_telemetry_report(self, tag, total_duration):
        """
        Processes dynamic logs, builds expressive console dashboards with 
        hierarchical sub-component tracking, and writes unified CSV ledgers.
        """
        metrics = self.telemetry.get(tag, {})
        if not metrics or not metrics.get("loss"):
            return
            
        # Update session tracking time directly
        metrics["total_duration"] = total_duration
        num_steps = len(metrics["loss"])

        # Identify all dynamically populated sub-loss fields (raw/weighted channels)
        baseline_keys = ["epoch", "batch", "global_step", "loss", "grad_norm", "lr", "vram_gb", "samples_per_sec", "total_duration"]
        dynamic_loss_keys = [k for k in metrics.keys() if isinstance(metrics[k], list) and k not in baseline_keys]

        # Shared math helper to extract metrics summary
        def stat_pack(data_list):
            return min(data_list), max(data_list), sum(data_list) / len(data_list), data_list[-1]

        # 📺 1. RENDER EXPRESSIVE CONSOLE MATRIX
        print("\n" + "═" * 110)
        print(f" 📊 PERFORMANCE RUN ANALYTICS LOG RECORD: [{tag.upper()}]")
        print("═" * 110)
        print(f"   • Total Run Wall Time:      {total_duration:.2f}s")
        print(f"   • Cumulative Graph Steps:   {num_steps}")
        print("-" * 110)
        print(f"   {'METRIC AXIS':<26} │ {'MINIMUM':<15} │ {'MAXIMUM':<15} │ {'AVERAGE':<15} │ {'TERMINAL OUT':<15}")
        print("-" * 110)

        # Unified line renderer handles dynamic text layouts and scientific format conversions
        def print_matrix_row(label, data, is_lr=False, unit=""):
            mn, mx, av, tm = stat_pack(data)
            if is_lr:
                print(f"   {label:<26} │ {mn:<15.2e} │ {mx:<15.2e} │ {av:<15.2e} │ {tm:<15.2e}")
            else:
                fmt = f"{{:.2f}}{unit}" if unit else "{:.4f}"
                print(f"   {label:<26} │ {fmt.format(mn):<15} │ {fmt.format(mx):<15} │ {fmt.format(av):<15} │ {fmt.format(tm):<15}")

        # Core Loss Track
        print_matrix_row("Loss (Total Combined)", metrics["loss"])
        
        # Dynamic Nested Loss Sub-Components Loop
        for k in sorted(dynamic_loss_keys):
            clean_label = f"  ↳ {k.replace('loss_', '').replace('_', ' ')}"
            print_matrix_row(clean_label, metrics[k])

        print("-" * 110)
        # Compute Resources & Hardware Infrastructure Metrics
        print_matrix_row("Gradient 2-Norm", metrics["grad_norm"])
        print_matrix_row("Learning Rate", metrics["lr"], is_lr=True)
        print_matrix_row("VRAM Memory Max", metrics["vram_gb"], unit=" GB")
        print_matrix_row("Throughput Velocity", metrics["samples_per_sec"], unit=" smpl/s")
        print("═" * 110 + "\n")

        # 💾 2. WRITE UNIFORM COLUMN LEDGER TO DISK (.CSV)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        csv_filename = os.path.join(self.cfg.checkpoint_dir, f"telemetry_{tag.lower().replace(' ', '_')}.csv")
        
        # Collect and normalize all target keys for column ordering
        ordered_keys = ["epoch", "batch", "global_step", "loss"] + sorted(dynamic_loss_keys) + ["grad_norm", "lr", "vram_gb", "samples_per_sec"]
        header_renames = {"lr": "learning_rate", "vram_gb": "max_vram_allocated_gb", "samples_per_sec": "throughput_samples_per_sec"}
        csv_headers = [header_renames.get(k, k) for k in ordered_keys]

        try:
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(csv_headers)
                
                for idx in range(num_steps):
                    row_data = []
                    for k in ordered_keys:
                        val = metrics[k][idx]
                        if k == "lr":
                            row_data.append(f"{val:.6e}")
                        elif isinstance(val, float):
                            row_data.append(f"{val:.6f}")
                        else:
                            row_data.append(val)  # Direct injection for step integers (epoch, batch, etc.)
                    writer.writerow(row_data)
            print(f"📌 [LEDGER EXPORTED SUCCESS] Telemetry statistics committed cleanly to -> {csv_filename}\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not write telemetry data to system log: {str(e)}")

    def dump_telemetry(self):
        for tag in self.used_tag:
            self._generate_and_save_telemetry_report(tag, self.telemetry[tag]["total_duration"])

    def _export_checkpoint(self, dict, name):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save(dict, checkpoint_path)
        print(f"📌 Checkpoint saved safely -> {checkpoint_path}")