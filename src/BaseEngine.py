import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import csv
import math

class BaseExecutionEngine:
    """
    🎯 CENTRAL RUNTIME ORCHESTRATOR: Highly scalable functional optimization loop.
    Upgraded with toggleable mixed precision configurations, dynamic AMP datatypes,
    precision-isolated reduction filters, and state checkpoint recovery loops.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.telemetry = {}
        self.used_tag = []
        
        # 🚀 AMP CONTROL MODULE: Dynamically sourced from configuration structures
        self.use_amp: bool = getattr(cfg, 'use_amp', True)
        self.amp_dtype: torch.dtype = getattr(cfg, 'amp_dtype', torch.bfloat16)
        
        # Scaler preserved for structural backward-compatibility if legacy float16 paths are called
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.use_amp and self.amp_dtype == torch.float16))

    def _compute_grad_norm(self, parameters):
        return sum(p.grad.detach().data.norm(2).item() ** 2 for p in parameters if p.grad is not None) ** 0.5
    
    def compute_alignment_loss(self, p_c, p_t, beta: float = 0.5):
        """
        🎯 STANDARDIZED SMOOTH L1 ALIGNMENT
        Standardizes both slot tensors to Mean=0, Std=1 across the feature 
        dimension prior to computing Smooth L1. Insulates the alignment 
        penalty from absolute weight-scale drift while preserving outlier robustness.
        """
        p_c = p_c.float()
        p_t = p_t.float()
        
        # 🚀 Standardize across the feature/channel dimension (dim=-1)
        mean_c = p_c.mean(dim=-1, keepdim=True)
        std_c = p_c.std(dim=-1, keepdim=True) + 1e-6
        p_c_std = (p_c - mean_c) / std_c
        
        mean_t = p_t.mean(dim=-1, keepdim=True)
        std_t = p_t.std(dim=-1, keepdim=True) + 1e-6
        p_t_std = (p_t - mean_t) / std_t
        
        # Bounded in a standard [Mean=0, Std=1] range. 
        # beta=0.5 now strictly translates to "transition at 0.5 standard deviations of error."
        loss_align = F.smooth_l1_loss(p_c_std, p_t_std, beta=beta)
        return loss_align

    def compute_variance_loss(self, z, target_std=1.0, eps=1e-4):
        z = z.float()
        if z.size(0) <= 1: return torch.tensor(0.0, device=z.device)
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(torch.clamp(target_std - std, min=0.0))

    def compute_covariance_loss(self, z):
        z = z.float()
        B, K, D = z.size()
        if B <= 1: 
            return torch.tensor(0.0, device=z.device)
        
        # 1. Center features across the batch dimension
        z_mean = z.mean(dim=0, keepdim=True)
        z_cent = z - z_mean
        
        # 2. Safe scale normalization via clamped standard deviation
        z_std = torch.sqrt(torch.sum(z_cent ** 2, dim=0, keepdim=True) / (B - 1) + 1e-8)
        z_std = torch.clamp(z_std, min=1e-2) 
        z_norm = z_cent / z_std
        
        # 3. Parallel Correlation matrix calculation [K, D, D] via BMM
        corr = torch.bmm(z_norm.permute(1, 2, 0), z_norm.permute(1, 0, 2)) / (B - 1)
        
        # 4. Mask out the diagonal elements
        diagonal_mask = torch.eye(D, device=z.device).unsqueeze(0) # [1, D, D]
        off_diag_corr = corr * (1.0 - diagonal_mask)
        
        # 🚀 THE MATHEMATICAL CORRECTION: Use squared L2 penalty (VICReg Standard)
        # divided by the exact count of off-diagonal elements.
        num_off_diagonals = K * D * (D - 1)
        loss = torch.sum(off_diag_corr ** 2) / num_off_diagonals
        
        return loss
    
    def compute_cross_slot_orthogonal_loss(self, z):
        z = z.float()
        B, K, D = z.size()
        if B <= 1: return torch.tensor(0.0, device=z.device)
        z_norm = torch.nn.functional.normalize(z, p=2, dim=-1)
        similarity = torch.bmm(z_norm, z_norm.transpose(1, 2))
        err = (similarity - torch.eye(K, device=z.device).unsqueeze(0).expand(B, -1, -1)) ** 2
        return err.sum() / (B * K * K)
    
    def create_warmup_cosine_scheduler(self, optimizer, num_warmup_steps: int, num_total_steps: int, min_lr_ratio: float = 0.0):
        def lr_lambda(step: int):
            if step < num_warmup_steps: return float(step) / float(max(1, num_warmup_steps))
            progress = float(step - num_warmup_steps) / float(max(1, num_total_steps - num_warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _execute_epoch_loop(self, tag, models, optimizer, data_loader, loss_fn_lambda, num_epochs=10, scheduler=None, before_step=None, after_step=None, after_epoch=None):
        self.used_tag.append(tag)
        
        session_file = os.path.join(self.cfg.checkpoint_dir, f"engine_state_{tag.lower().replace(' ', '_')}.pt")
        
        resume_session = False
        start_epoch = 0
        start_batch = -1
        global_step_idx = 0
        accumulated_duration = 0.0
        
        # 🚀 DYNAMIC UPGRADE: Linked precision target profile directly to config variables
        target_dtype = self.amp_dtype
        use_scaler = self.use_amp and target_dtype == torch.float16

        if os.path.exists(session_file):
            try:
                ckpt = torch.load(session_file, map_location=self.device)
                if ckpt.get('completed', False) or ckpt.get('epoch', 0) >= num_epochs:
                    print(f"✨ [AUTO-RETRAIN] Completed checkpoint found for [{tag.upper()}] ({ckpt.get('epoch')}/{num_epochs} epochs). Re-initializing weights to retrain from scratch.")
                else:
                    print(f"🔄 [AUTO-RESUME] Unfinished session detected for [{tag.upper()}]. Restoring tracking matrices and resuming from Epoch {ckpt['epoch'] + 1}, Batch {ckpt.get('batch_idx', 0) + 1}.")
                    
                    for idx, m in enumerate(models):
                        m.load_state_dict(ckpt['model_states'][idx])
                    
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                    if scheduler is not None and ckpt.get('scheduler_state') is not None:
                        scheduler.load_state_dict(ckpt['scheduler_state'])
                    if ckpt.get('scaler_state') is not None and use_scaler:
                        self.scaler.load_state_dict(ckpt['scaler_state'])
                        
                    start_epoch = ckpt['epoch']
                    start_batch = ckpt.get('batch_idx', -1)
                    global_step_idx = ckpt['global_step_idx']
                    accumulated_duration = ckpt.get('accumulated_duration', 0.0)
                    self.telemetry[tag] = ckpt['telemetry_snapshot']
                    
                    # 🛡️ TRUNCATION FILTER: Slice tracking logs to prevent duplication pollution
                    metrics = self.telemetry[tag]
                    for metric_key in list(metrics.keys()):
                        if isinstance(metrics[metric_key], list):
                            metrics[metric_key] = metrics[metric_key][:global_step_idx]
                            
                    resume_session = True
            except Exception as e:
                print(f"⚠️ Checkpoint file corrupted ({str(e)}). Initializing training from scratch.")

        if not resume_session:
            self.telemetry[tag] = {
                "epoch": [], "batch": [], "global_step": [],
                "loss": [], "grad_norm": [], "lr": [],
                "vram_gb": [], "samples_per_sec": [],
                "total_duration": 0.0
            }
        
        metrics = self.telemetry[tag]
        trainable_params = [p for m in models for p in m.parameters() if p.requires_grad]
        
        print(f"🚀 Initiating High-Order Optimization Pass: [{tag.upper()}] | Budget: {num_epochs} Epochs")
        print(f"⚙️ Precision Settings: AMP Enabled={self.use_amp} | Target Datatype={target_dtype}")
        loop_start_time = time.perf_counter()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            
            for batch_idx, batch in enumerate(data_loader):
                # 🤝 BATCH-LEVEL RESUMPTION FAST-FORWARD GAP FILTER
                if epoch == start_epoch and batch_idx <= start_batch:
                    continue
                    
                batch_start = time.perf_counter()
                optimizer.zero_grad()
                batch_size = batch['feature_ids'].size(0) if 'feature_ids' in batch else self.cfg.batch_size
                
                # 🚀 UPGRADE: Leveraged 'enabled' attribute to pass config context without nested logic blocks
                with torch.amp.autocast('cuda', dtype=target_dtype, enabled=self.use_amp):
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
                            if k != "total":
                                component_logs.append(f"{short_name}:{raw_loss.item():.3f}")

                # 🚀 UPGRADE: Standardized backpropagation to bypass execution scaling if running bfloat16 or standard float32
                if not use_scaler:
                    total_loss.backward()
                    grad_norm = self._compute_grad_norm(trainable_params)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip_norm)
                    optimizer.step()
                else:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(optimizer)
                    grad_norm = self._compute_grad_norm(trainable_params)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                
                if scheduler is not None:
                    scheduler.step()

                if after_step is not None:
                    after_step()
                    
                step_duration = time.perf_counter() - batch_start
                samples_per_sec = batch_size / step_duration if step_duration > 0 else 0.0
                
                current_lr = optimizer.param_groups[0]['lr']
                vram_usage_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                
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
                        f"LR: {current_lr:.1e} │ {vram_usage_gb:.1f}GB │ {samples_per_sec:.0f}sam/s │ "
                        f"🧬 {comp_str}"
                    )
                    
                    # 💾 INTRA-EPOCH MITIGATION SAVER: Safely captures logs during unexpected failures
                    active_runtime_duration = accumulated_duration + (time.perf_counter() - loop_start_time)
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'global_step_idx': global_step_idx,
                        'accumulated_duration': active_runtime_duration,
                        'model_states': [m.state_dict() for m in models],
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                        'scaler_state': self.scaler.state_dict() if use_scaler else None,
                        'telemetry_snapshot': metrics,
                        'completed': False
                    }, session_file)
                          
            print(f"--- [{tag}] Epoch {epoch+1:02d} Complete | Duration: {time.perf_counter() - epoch_start:.2f}s ---")
            
            # Reset active batch window position pointer for clean sequential starts on subsequent loops
            start_batch = -1
            
            # 🛑 EARLY STOPPING LIFE-CYCLE BREAKOUT DETECTOR
            stop_early_triggered = False
            if after_epoch is not None:
                stop_early_triggered = after_epoch(epoch + 1)
                
            active_runtime_duration = accumulated_duration + (time.perf_counter() - loop_start_time)
            
            if stop_early_triggered:
                print(f"🛑 [EARLY STOP BREAKOUT] Terminating optimized epoch tracking. Saving unified persistent memory blocks.")
                torch.save({
                    'epoch': epoch + 1,
                    'batch_idx': -1,
                    'global_step_idx': global_step_idx,
                    'accumulated_duration': active_runtime_duration,
                    'model_states': [m.state_dict() for m in models],
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                    'scaler_state': self.scaler.state_dict() if use_scaler else None,
                    'telemetry_snapshot': metrics,
                    'completed': True 
                }, session_file)
                break
                
            torch.save({
                'epoch': epoch + 1,
                'batch_idx': -1,
                'global_step_idx': global_step_idx,
                'accumulated_duration': active_runtime_duration,
                'model_states': [m.state_dict() for m in models],
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'scaler_state': self.scaler.state_dict() if use_scaler else None,
                'telemetry_snapshot': metrics,
                'completed': (epoch + 1) >= num_epochs
            }, session_file)
            
        total_duration = (time.perf_counter() - loop_start_time) + accumulated_duration
        metrics["total_duration"] = total_duration

    def _generate_and_save_telemetry_report(self, tag, total_duration):
        metrics = self.telemetry.get(tag, {})
        if not metrics or not metrics.get("loss"): return
        metrics["total_duration"] = total_duration
        num_steps = len(metrics["loss"])

        baseline_keys = ["epoch", "batch", "global_step", "loss", "grad_norm", "lr", "vram_gb", "samples_per_sec", "total_duration"]
        dynamic_loss_keys = [k for k in metrics.keys() if isinstance(metrics[k], list) and k not in baseline_keys]

        def stat_pack(data_list):
            return min(data_list), max(data_list), sum(data_list) / len(data_list), data_list[-1]

        print("\n" + "═" * 110)
        print(f" 📊 PERFORMANCE RUN ANALYTICS LOG RECORD: [{tag.upper()}]")
        print("═" * 110)
        print(f"   • Total Run Wall Time:      {total_duration:.2f}s")
        print(f"   • Cumulative Graph Steps:   {num_steps}")
        print("-" * 110)
        print(f"   {'METRIC AXIS':<26} │ {'MINIMUM':<15} │ {'MAXIMUM':<15} │ {'AVERAGE':<15} │ {'TERMINAL OUT':<15}")
        print("-" * 110)

        def print_matrix_row(label, data, is_lr=False, unit=""):
            mn, mx, av, tm = stat_pack(data)
            if is_lr:
                print(f"   {label:<26} │ {mn:<15.2e} │ {mx:<15.2e} │ {av:<15.2e} │ {tm:<15.2e}")
            else:
                fmt = f"{{:.2f}}{unit}" if unit else "{:.4f}"
                print(f"   {label:<26} │ {fmt.format(mn):<15} │ {fmt.format(mx):<15} │ {fmt.format(av):<15} │ {fmt.format(tm):<15}")

        print_matrix_row("Loss (Total Combined)", metrics["loss"])
        for k in sorted(dynamic_loss_keys):
            print_matrix_row(f"  ↳ {k.replace('loss_', '').replace('_', ' ')}", metrics[k])

        print("-" * 110)
        print_matrix_row("Gradient 2-Norm", metrics["grad_norm"])
        print_matrix_row("Learning Rate", metrics["lr"], is_lr=True)
        print_matrix_row("VRAM Memory Max", metrics["vram_gb"], unit=" GB")
        print_matrix_row("Throughput Velocity", metrics["samples_per_sec"], unit=" smpl/s")
        print("═" * 110 + "\n")

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        csv_filename = os.path.join(self.cfg.checkpoint_dir, f"telemetry_{tag.lower().replace(' ', '_')}.csv")
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
                        if k == "lr": row_data.append(f"{val:.6e}")
                        elif isinstance(val, float): row_data.append(f"{val:.6f}")
                        else: row_data.append(val)
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