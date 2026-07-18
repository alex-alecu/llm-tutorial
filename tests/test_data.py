from pathlib import Path

import numpy as np

from macllm.data import TokenDataset


def test_token_batch_is_shifted_by_one(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint32).tofile(path)
    dataset = TokenDataset(path)
    inputs, targets = dataset.batch(np.random.default_rng(2), 3, 8)
    assert inputs.shape == (3, 8)
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
