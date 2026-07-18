from macllm.config import PRESETS, get_preset


def test_presets_are_in_increasing_order():
    assert list(PRESETS) == ["quick", "standard", "overnight"]
    assert PRESETS["quick"].model.dim < PRESETS["standard"].model.dim
    assert PRESETS["standard"].model.dim < PRESETS["overnight"].model.dim


def test_all_model_shapes_are_valid():
    for name in PRESETS:
        model = get_preset(name).model
        model.validate()
        assert model.dim % 64 == 0
        assert model.hidden_dim % 64 == 0
