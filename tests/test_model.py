import mlx.core as mx
import numpy as np

from macllm.config import ModelConfig
from macllm.model import TinyLM, parameter_count


def small_model() -> TinyLM:
    config = ModelConfig(
        vocab_size=64,
        dim=64,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
    )
    model = TinyLM(config)
    mx.eval(model.parameters())
    return model


def test_model_shapes_and_cache():
    model = small_model()
    logits, cache = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64)
    assert len(cache) == 2
    next_logits, next_cache = model(mx.array([[5]]), cache)
    assert next_logits.shape == (1, 1, 64)
    assert next_cache[0][0].shape[2] == 5
    assert parameter_count(model) > 0


def test_future_tokens_do_not_change_past_logits():
    model = small_model()
    first, _ = model(mx.array([[1, 2, 3, 4]]))
    second, _ = model(mx.array([[1, 2, 8, 9]]))
    np.testing.assert_allclose(
        np.array(first[:, :2, :]), np.array(second[:, :2, :]), rtol=1e-4, atol=1e-4
    )
