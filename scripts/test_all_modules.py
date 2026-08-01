# scripts/test_all_modules.py
import gc
import time
import json
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CardioConfig
from Pipeline import ClinicalPipeline
from src.ModelModules import (
    UnifiedSystemicEmbedder,
    PerceiverLatentPooling,
    LinearProbeHead,
    LabelAttentiveProbe,
)

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def create_benchmark_batch(
    cfg: CardioConfig, 
    benchmark_bs: int, 
    num_feats: int, 
    num_cats: int,
    num_cls: int
) -> Dict[str, Any]:
    L = cfg.max_sequence_len
    subwords = cfg.max_subwords
    device = cfg.device
    return {
        'patient_session_id': [f"pat_{i}" for i in range(benchmark_bs)],
        'feature_ids': torch.randint(0, num_feats, (benchmark_bs, L), device=device),
        'numeric_values': torch.randn(benchmark_bs, L, device=device),
        'cat_result_ids': torch.randint(0, num_cats, (benchmark_bs, L, subwords), device=device),
        'timestamps': torch.sort(torch.rand(benchmark_bs, L, device=device) * 100.0, dim=-1, descending=True)[0],
        'base_mask': torch.zeros(benchmark_bs, L, dtype=torch.bool, device=device),
        'student_mask': torch.zeros(benchmark_bs, L, dtype=torch.bool, device=device),
        'teacher_mask': torch.zeros(benchmark_bs, L, dtype=torch.bool, device=device),
        'dt_target': torch.rand(benchmark_bs, device=device) * 24.0,
        'age': torch.rand(benchmark_bs, device=device),
        'gender': torch.randint(1, 3, (benchmark_bs,), device=device),
        'icd_targets': torch.randint(0, num_cls, (benchmark_bs, cfg.max_targets), device=device),
        'target_mask': torch.zeros(benchmark_bs, cfg.max_targets, dtype=torch.bool, device=device)
    }


class VRAMFootprintProfiler:
    def __init__(self, cfg: CardioConfig, benchmark_bs: Optional[int] = None):
        self.cfg = cfg
        self.device = cfg.device
        self.use_amp = cfg.use_amp
        self.amp_dtype = cfg.amp_dtype
        self.benchmark_bs = benchmark_bs if benchmark_bs is not None else cfg.batch_size

        with open(cfg.codebook_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)["metadata"]

        self.num_feats = meta['num_total_features']
        self.num_cats = meta['num_cat_results']
        self.num_cls = meta['num_icd_classes']
        
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            alloc_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            print(f"🖥️ GPU Total Capacity: {total_mem:.2f} GB │ Background Allocated: {alloc_mem:.2f} GB")
            if alloc_mem > 2.0 and self.benchmark_bs == self.cfg.batch_size:
                self.benchmark_bs = min(64, self.benchmark_bs)

        self.batch = create_benchmark_batch(
            self.cfg, 
            benchmark_bs=self.benchmark_bs,
            num_feats=self.num_feats,
            num_cats=self.num_cats,
            num_cls=self.num_cls
        )

    def get_static_memory_stats(self, module: Optional[nn.Module]) -> Tuple[int, int, float]:
        if module is None or not isinstance(module, nn.Module): return 0, 0, 0.0
        tot_params = sum(p.numel() for p in module.parameters())
        trn_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
        return tot_params, trn_params, (param_bytes + buffer_bytes) / (1024 ** 2)

    def profile_execution(self, mod_fn: Any, inputs: tuple, enable_grad: bool = False, run_backward: bool = False) -> Tuple[float, float]:
        if not torch.cuda.is_available(): return 0.0, 0.0

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        start_alloc = torch.cuda.memory_allocated(self.device)
        start_time = time.perf_counter()
        
        out, loss = None, None
        try:
            with torch.set_grad_enabled(enable_grad):
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    out = mod_fn(*inputs)
                    if isinstance(out, dict):
                        loss = out['logits'].sum() if ('logits' in out and out['logits'] is not None) else out['z_c_slots'].sum()
                    elif isinstance(out, tuple): loss = out[0].sum()
                    else: loss = out.sum()

                if enable_grad and run_backward: loss.backward()

            torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            dynamic_peak_mb = max(0.0, (peak_alloc - start_alloc) / (1024 ** 2))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return -1.0, -1.0
        finally:
            del out, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        return dynamic_peak_mb, elapsed_ms

    def run_full_benchmark(self):
        print("\n" + "═"*105)
        print(f"📊 CHRONOS-JEPA VRAM FOOTPRINT REPORT (Device: {self.device} | AMP: {self.use_amp})")
        print(f"   Benchmark Batch Size: {self.benchmark_bs} │ Sequence Length: {self.cfg.max_sequence_len}")
        print("═"*105)

        pipeline = ClinicalPipeline(self.cfg, self.device)
        pipeline.inject_phase2_infrastructure()

        D, K, K_aug = self.cfg.latent_dim, self.cfg.num_slots, pipeline.augmented_slots

        tokenizer      = UnifiedSystemicEmbedder(self.num_feats, self.num_cats, d_model=D, num_frequencies=self.cfg.fourier_frequencies).to(self.device)
        perceiver      = PerceiverLatentPooling(num_slots=K, d_model=D).to(self.device)
        probe_linear   = LinearProbeHead(in_slots=K_aug, in_dim=D, num_classes=self.num_cls, dropout_p=self.cfg.probe_dropout).to(self.device)
        probe_laat     = LabelAttentiveProbe(in_dim=D, num_classes=self.num_cls, dropout_p=self.cfg.probe_dropout).to(self.device)

        f_ids  = self.batch['feature_ids'].to(self.device)
        v_nums = self.batch['numeric_values'].to(self.device)
        c_ids  = self.batch['cat_result_ids'].to(self.device)
        times  = self.batch['timestamps'].to(self.device)
        s_mask = self.batch['student_mask'].to(self.device)
        age    = self.batch['age'].to(self.device).float()
        gender = self.batch['gender'].to(self.device).long()

        with torch.no_grad():
            tok_out = tokenizer(f_ids, v_nums, c_ids, times)
            z_c_raw = pipeline.context_encoder(f_ids, v_nums, c_ids, times, s_mask)
            z_final = pipeline.assembler(z_c_raw, age, gender)

        modules_to_test = [
            ("Unified Systemic Tokenizer", tokenizer, (f_ids, v_nums, c_ids, times)),
            ("Perceiver Latent Pooling", perceiver, (tok_out, s_mask)),
            ("Context Encoder Backbone", pipeline.context_encoder, (f_ids, v_nums, c_ids, times, s_mask)),
            ("JEPA Predictor Network", pipeline.predictor, (z_c_raw,)),
            ("Manifold Assembler", pipeline.assembler, (z_c_raw, age, gender)),
            ("Hopfield Memory", pipeline.hopfield, (z_final,)),
            ("Label Attentive Probe", probe_laat, (z_final,)),
            ("Linear Probe Head", probe_linear, (z_final,)),
            ("Auxiliary Cardinality Head", pipeline.cardinal, (z_final,)),
            ("E2E Phase 1 Pretraining Pass", lambda b: pipeline.process_batch(b, self.device, run_teacher=True), (self.batch,)),
            ("E2E Phase 2 Probing Pass", lambda b: pipeline.process_batch(b, self.device, run_teacher=False), (self.batch,))
        ]

        print("\n📦 1. MODULE STATIC WEIGHT & BUFFER VRAM BREAKDOWN:")
        print("─"*105)
        print(f"   {'MODULE NAME':<42} │ {'TOTAL PARAMS':<14} │ {'TRAINABLE':<14} │ {'WEIGHT VRAM':<12}")
        print("─"*105)

        tot_all_params, trn_all_params, static_vram_sum = 0, 0, 0.0
        for name, mod, _ in modules_to_test:
            if callable(mod) and not isinstance(mod, nn.Module):
                continue
            tot_p, trn_p, mb = self.get_static_memory_stats(mod)
            tot_all_params += tot_p
            trn_all_params += trn_p
            static_vram_sum += mb
            print(f"   • {name:<40} │ Total: {tot_p:12,} │ Trainable: {trn_p:12,} │ VRAM: {mb:8.2f} MB")

        print("─"*105)
        print(f"   STATIC SUMMARY TOTALS                      │ {tot_all_params:12,} │ {trn_all_params:12,} │ VRAM: {static_vram_sum:8.2f} MB")
        print("═"*105)

        print("\n⚡ 2. DYNAMIC FORWARD & BACKWARD PASS VRAM PROFILING:")
        print("─"*105)
        print(f"   {'MODULE NAME':<42} │ {'INFERENCE (NO-GRAD)':<18} │ {'TRAIN (FWD ONLY)':<18} │ {'TRAIN (FWD+BWD)':<18}")
        print(f"   {'':<42} │ {'VRAM':<8} {'TIME':<8} │ {'VRAM':<8} {'TIME':<8} │ {'VRAM':<8} {'TIME':<8}")
        print("─"*105)

        for name, mod, args in modules_to_test:
            mod_fn = mod.forward if isinstance(mod, nn.Module) else mod
            m_nograd, t_nograd = self.profile_execution(mod_fn, args, enable_grad=False, run_backward=False)
            m_fwd, t_fwd       = self.profile_execution(mod_fn, args, enable_grad=True, run_backward=False)
            m_bwd, t_bwd       = self.profile_execution(mod_fn, args, enable_grad=True, run_backward=True)
            fmt = lambda m, t: f"{m:6.1f}MB {t:6.1f}ms" if m >= 0 else "  [OOM ERROR] "
            print(f"   • {name:<40} │ {fmt(m_nograd, t_nograd)} │ {fmt(m_fwd, t_fwd)} │ {fmt(m_bwd, t_bwd)}")

        print("═"*105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = CardioConfig()
    profiler = VRAMFootprintProfiler(cfg, benchmark_bs=args.batch_size)
    profiler.run_full_benchmark()