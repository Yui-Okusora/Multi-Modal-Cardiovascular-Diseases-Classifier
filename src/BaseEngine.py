# src/BaseEngine.py
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
    Supports mixed precision, dynamic AMP datatypes, precision-isolated reduction filters,
    gradient accumulation (decoupled physical & effective batch sizes), and full telemetry 
    snapshot reporting with safe state-resumption.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.telemetry = {}
        self.used_tag = []
        
        self.use_amp: bool = getattr(cfg, 'use_amp', True)
        self.amp_dtype: torch.dtype = getattr(cfg, 'amp_dtype', torch.bfloat16)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.use_amp and self.amp_dtype == torch.float16))

    def _compute_grad_norm(self, parameters):
        return sum(p.grad.detach().data.norm(2).item() ** 2 for p in parameters if p.grad is not None) ** 0.5
    
    def compute_alignment_loss(self, p_c, p_t, beta: float = 0.5):
        p_c, p_t = p_c.float(), p_t.float()
        mean_c, std_c = p_c.mean(dim=-1, keepdim=True), p_c.std(dim=-1, keepdim=True) + 1e-6
        p_c_std = (p_c - mean_c) / std_c
        
        mean_t, std_t = p_t.mean(dim=-1, keepdim=True), p_t.std(dim=-1, keepdim=True) + 1e-6
        p_t_std = (p_t - mean_t) / std_t
        return F.smooth_l1_loss(p_c_std, p_t_std, beta=beta)

    def compute_variance_loss(self, z, target_std=1.0, eps=1e-4):
        z = z.float()
        if z.size(0) <= 1: return torch.tensor(0.0, device=z.device)
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(torch.clamp(target_std - std, min=0.0))

    def compute_covariance_loss(self, z):
        z = z.float()
        B, K, D = z.size()
        if B <= 1: return torch.tensor(0.0, device=z.device)
        
        z_cent = z - z.mean(dim=0, keepdim=True)
        z_std = torch.clamp(torch.sqrt(torch.sum(z_cent ** 2, dim=0, keepdim=True) / (B - 1) + 1e-8), min=1e-2)
        z_norm = z_cent / z_std
        
        corr = torch.bmm(z_norm.permute(1, 2, 0), z_norm.permute(1, 0, 2)) / (B - 1)
        diagonal_mask = torch.eye(D, device=z.device).unsqueeze(0)
        off_diag_corr = corr * (1.0 - diagonal_mask)
        return torch.sum(off_diag_corr ** 2) / (K * D * (D - 1))
    
    def compute_cross_slot_orthogonal_loss(self, z):
        z = z.float()
        B, K, D = z.size()
        if B <= 1: return torch.tensor(0.0, device=z.device)
        z_norm = F.normalize(z, p=2, dim=-1)
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
        
        # -------------------------------------------------------------
        # 🚀 GRADIENT ACCUMULATION RESOLUTION
        # -------------------------------------------------------------
        physical_bs = getattr(self.cfg, 'batch_size', 32)
        grad_accum_steps = getattr(self.cfg, 'grad_accum_steps', None)
        if grad_accum_steps is None:
            effective_bs = getattr(self.cfg, 'effective_batch_size', physical_bs)
            accum_steps = max(1, effective_bs // physical_bs)
        else:
            accum_steps = max(1, grad_accum_steps)
            effective_bs = physical_bs * accum_steps

        # Align log interval to nearest multiple of accum_steps so logs always land on update steps
        raw_log_interval = getattr(self.cfg, 'log_interval', 50)
        aligned_log_interval = max(accum_steps, (raw_log_interval // accum_steps) * accum_steps)

        resume_session = False
        start_epoch, start_batch, global_step_idx, accumulated_duration = 0, -1, 0, 0.0
        target_dtype, use_scaler = self.amp_dtype, (self.use_amp and self.amp_dtype == torch.float16)

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
                    
                    start_epoch, start_batch = ckpt['epoch'], ckpt.get('batch_idx', -1)
                    global_step_idx, accumulated_duration = ckpt['global_step_idx'], ckpt.get('accumulated_duration', 0.0)
                    self.telemetry[tag] = ckpt['telemetry_snapshot']
                    
                    # 🛡️ TRUNCATION FILTER: Slice tracking logs to prevent duplication pollution
                    metrics = self.telemetry[tag]
                    for metric_key in list(metrics.keys()):
                        if isinstance(metrics[metric_key], list):
                            metrics[metric_key] = metrics[metric_key][:global_step_idx]
                            
                    resume_session = True
            except Exception as e:
                print(f"⚠️ Checkpoint corrupted ({str(e)}). Training from scratch.")

        if not resume_session:
            self.telemetry[tag] = {
                "epoch": [], "batch": [], "global_step": [], "loss": [], 
                "grad_norm": [], "lr": [], "vram_gb": [], "samples_per_sec": [], "total_duration": 0.0
            }
        
        metrics = self.telemetry[tag]
        trainable_params = [p for m in models for p in m.parameters() if p.requires_grad]
        
        print(f"🚀 Initiating High-Order Optimization Pass: [{tag.upper()}] | Budget: {num_epochs} Epochs")
        print(f"⚙️ Precision Settings: AMP Enabled={self.use_amp} | Target Datatype={target_dtype}")
        print(f"📦 Batch Config: Physical={physical_bs} | Effective={effective_bs} | Accumulation Steps={accum_steps}")
        
        loop_start_time = time.perf_counter()
        last_grad_norm = 0.0
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.perf_counter()
            optimizer.zero_grad()  # Initialize gradients for epoch

            for batch_idx, batch in enumerate(data_loader):
                if epoch == start_epoch and batch_idx <= start_batch: 
                    continue
                if torch.cuda.is_available(): 
                    torch.cuda.reset_peak_memory_stats(self.device)
                    
                batch_start = time.perf_counter()
                current_batch_size = batch['feature_ids'].size(0) if 'feature_ids' in batch else physical_bs
                
                # Check if this physical batch triggers an optimizer step or is end-of-epoch flush
                is_update_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(data_loader))
                is_log_batch = ((batch_idx + 1) % aligned_log_interval == 0) or (batch_idx + 1 == len(data_loader))
                
                # -------------------------------------------------------------
                # 1. FORWARD PASS & UN-SCALED LOSS LOGGING
                # -------------------------------------------------------------
                with torch.amp.autocast('cuda', dtype=target_dtype, enabled=self.use_amp):
                    loss_output = loss_fn_lambda(batch, global_step_idx, len(data_loader) * num_epochs)
                    total_loss, component_logs = 0.0, []
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

                # -------------------------------------------------------------
                # 2. NORMALIZED BACKWARD PASS (Prevent gradient scaling explosion)
                # -------------------------------------------------------------
                loss_to_backward = total_loss / accum_steps

                if not use_scaler:
                    loss_to_backward.backward()
                else:
                    self.scaler.scale(loss_to_backward).backward()

                # -------------------------------------------------------------
                # 3. CONDITIONAL OPTIMIZER STEP & GRADIENT CLIPPING
                # -------------------------------------------------------------
                if is_update_step:
                    if not use_scaler:
                        last_grad_norm = self._compute_grad_norm(trainable_params)
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip_norm)
                        optimizer.step()
                    else:
                        self.scaler.unscale_(optimizer)
                        last_grad_norm = self._compute_grad_norm(trainable_params)
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.grad_clip_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()

                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                    if after_step is not None: 
                        after_step()
                    
                step_duration = time.perf_counter() - batch_start
                samples_per_sec = current_batch_size / step_duration if step_duration > 0 else 0.0
                current_lr = optimizer.param_groups[0]['lr']
                vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                
                # -------------------------------------------------------------
                # 4. TELEMETRY RECORDING
                # -------------------------------------------------------------
                metrics["epoch"].append(epoch + 1)
                metrics["batch"].append(batch_idx)
                metrics["global_step"].append(global_step_idx)
                metrics["loss"].append(total_loss.item())  # Log exact unscaled physical loss
                metrics["grad_norm"].append(last_grad_norm)
                metrics["lr"].append(current_lr)
                metrics["vram_gb"].append(vram_gb)
                metrics["samples_per_sec"].append(samples_per_sec)
                global_step_idx += 1
                
                if is_log_batch:
                    comp_str = " ｜ ".join(component_logs) if component_logs else "No active sub-components"
                    accum_tag = "⚡ STEP" if is_update_step else "⏳ ACCUM"
                    print(
                        f"{accum_tag} [{tag:<12}] E{epoch+1:02d} B{batch_idx:03d}/{len(data_loader):03d} │ "
                        f"L_tot: {total_loss.item():.4f} │ G: {last_grad_norm:5.2f} │ "
                        f"LR: {current_lr:.1e} │ {vram_gb:.1f}GB │ {samples_per_sec:.0f}sam/s │ 🧬 {comp_str}"
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
            
            start_batch = -1
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
        """
        📊 FULL PERFORMANCE RUN ANALYTICS LOG RECORD & CSV EXPORT
        """
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
