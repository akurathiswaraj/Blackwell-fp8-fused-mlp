import torch
from torch import nn

from integration.fp8_mlp import (
    dispatch_mlp,
    interleave_gate_up_weight,
)

FP8_MAX = 448.0

class BlackwellFP8LlamaMLP(nn.Module):
    """Llama SwiGLU MLP backed by the SM100 direct epilogue.

    The fast path accepts prequantized FP8 activations through
    forward_fp8(). forward() additionally performs dynamic,
    per-tensor activation quantization for functional integration.
    """

    def __init__(
        self,
        blocked_gate_up_weight,
        down_weight,
        scale_gate_up,
        scale_down,
        quant_multiplier,
        direct_extension,
        down_extension,
    ):
        super().__init__()

        if blocked_gate_up_weight.dim() != 2:
            raise ValueError(
                "gate/up weight must have shape [2N,K]"
            )

        if down_weight.dim() != 2:
            raise ValueError(
                "down weight must have shape [K,N]"
            )

        hidden_size = blocked_gate_up_weight.shape[1]
        intermediate_size = (
            blocked_gate_up_weight.shape[0] // 2
        )

        if down_weight.shape != (
            hidden_size,
            intermediate_size,
        ):
            raise ValueError(
                "down weight shape does not match "
                "gate/up dimensions"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_multiplier = float(
            quant_multiplier
        )

        self.register_buffer(
            "gate_up_weight",
            interleave_gate_up_weight(
                blocked_gate_up_weight
            ),
            persistent=True,
        )
        self.register_buffer(
            "down_weight",
            down_weight.contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "scale_gate_up",
            torch.as_tensor(
                scale_gate_up,
                device=blocked_gate_up_weight.device,
                dtype=torch.float32,
            ).reshape(()),
            persistent=True,
        )
        self.register_buffer(
            "scale_down",
            torch.as_tensor(
                scale_down,
                device=down_weight.device,
                dtype=torch.float32,
            ).reshape(()),
            persistent=True,
        )
        self.register_buffer(
            "running_amax",
            torch.zeros(
                1,
                device=down_weight.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Extension modules are runtime objects, not state_dict data.
        object.__setattr__(
            self,
            "_direct_extension",
            direct_extension,
        )
        object.__setattr__(
            self,
            "_down_extension",
            down_extension,
        )

    def reset_amax(self):
        self.running_amax.zero_()

    def forward_fp8(
        self,
        activation_fp8,
        scale_activation,
    ):
        if (
            activation_fp8.dtype
            != torch.float8_e4m3fn
        ):
            raise TypeError(
                "forward_fp8 expects float8_e4m3fn"
            )

        if (
            activation_fp8.shape[-1]
            != self.hidden_size
        ):
            raise ValueError(
                "activation hidden dimension mismatch"
            )

        original_shape = activation_fp8.shape
        flat = activation_fp8.reshape(
            -1,
            self.hidden_size,
        ).contiguous()

        scale_activation = torch.as_tensor(
            scale_activation,
            device=flat.device,
            dtype=torch.float32,
        ).reshape(())

        output = dispatch_mlp(
            flat,
            self.gate_up_weight,
            self.down_weight,
            scale_activation,
            self.scale_gate_up,
            self.scale_down,
            self.quant_multiplier,
            self.running_amax,
            self._direct_extension,
            self._down_extension,
        )

        return output.reshape(
            *original_shape[:-1],
            self.hidden_size,
        )

    def forward(self, activation):
        if (
            activation.shape[-1]
            != self.hidden_size
        ):
            raise ValueError(
                "activation hidden dimension mismatch"
            )

        activation_f32 = activation.float()
        activation_amax = (
            activation_f32.abs().amax()
        )

        scale_activation = (
            activation_amax / FP8_MAX
        ).clamp_min(
            torch.finfo(torch.float32).tiny
        )

        activation_fp8 = torch.clamp(
            activation_f32 / scale_activation,
            -FP8_MAX,
            FP8_MAX,
        ).to(torch.float8_e4m3fn)

        return self.forward_fp8(
            activation_fp8,
            scale_activation,
        )
