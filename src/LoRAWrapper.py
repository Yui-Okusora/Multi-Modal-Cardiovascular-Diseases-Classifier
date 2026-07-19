# src/LoRAWrapper.py
import torch
import torch.nn as nn
import math
import types

class LoRALinearWrapper(nn.Module):
    """🎯 SINGULAR LOW-RANK ADAPTER CHANNEL"""
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        self.rank = rank
        self.scaling = alpha / rank

        # Pristine 2D Parameter Matrices
        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_outputs = self.base_layer(x)
        lora_outputs = (x @ self.lora_A.T) @ self.lora_B.T
        return base_outputs + (lora_outputs * self.scaling)

    def defactorize_and_merge(self) -> nn.Linear:
        """Bakes low-rank delta updates back into the base layer weight matrix."""
        with torch.no_grad():
            weight_delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.add_(weight_delta.to(self.base_layer.weight.device))
        return self.base_layer

def defactorize_entire_architecture(module: nn.Module):
    """Recursively scans the graph to merge adapters and restore native nn.Linear layers."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinearWrapper):
            native_linear = child.defactorize_and_merge()
            setattr(module, name, native_linear)
        else:
            defactorize_entire_architecture(child)

def clean_transformer_layer_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    if getattr(self, "norm_first", False):
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
        x = self.norm2(x + self._ff_block(x))
    return x

def inject_lora_infrastructure(module: nn.Module, rank: int = 16, alpha: float = 32.0, target_names=None):
    """Recursively injects standalone adapters into target layers"""
    if target_names is None:
        target_names = ["linear1", "linear2", "channel_mlp", "slot_combiner"]

    if isinstance(module, nn.TransformerEncoderLayer):
        module.forward = types.MethodType(clean_transformer_layer_forward, module)

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t_name in name for t_name in target_names):
            target_device = child.weight.device
            wrapper = LoRALinearWrapper(child, rank=rank, alpha=alpha).to(target_device)
            
            if not hasattr(wrapper, "weight"): type(wrapper).weight = property(lambda self: self.base_layer.weight)
            if not hasattr(wrapper, "bias"): type(wrapper).bias = property(lambda self: self.base_layer.bias)
            setattr(module, name, wrapper)
        else:
            inject_lora_infrastructure(child, rank, alpha, target_names)