"""Approximately one-billion-parameter Llama-style proxy model."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from transformers import LlamaConfig, LlamaForCausalLM


@dataclass(frozen=True)
class ProxyModelConfig:
    # With vocab_size=196,608 and tied embeddings this is ~996.4M params.
    hidden_size: int = 1792
    intermediate_size: int = 4864
    num_hidden_layers: int = 18
    num_attention_heads: int = 14
    num_key_value_heads: int = 7
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0


def build_proxy_model(
    *,
    vocab_size: int,
    block_size: int,
    bos_token_id: int | None,
    eos_token_id: int,
    pad_token_id: int,
    model_config: ProxyModelConfig | None = None,
) -> LlamaForCausalLM:
    spec = model_config or ProxyModelConfig()
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=block_size,
        rms_norm_eps=spec.rms_norm_eps,
        rope_theta=spec.rope_theta,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=True,
        use_cache=False,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.config.use_cache = False
    return model


def parameter_report(model: LlamaForCausalLM) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "parameters_billions": total / 1_000_000_000,
    }
