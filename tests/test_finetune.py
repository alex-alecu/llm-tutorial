from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from macllm.finetune import make_batch


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Question: hello Answer: world"],
        trainers.BpeTrainer(vocab_size=64, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]),
    )
    return tokenizer


def test_prompt_tokens_are_masked():
    _, _, mask = make_batch(
        _tokenizer(),
        [{"prompt": "Question: hello Answer:", "completion": " world"}],
        [0],
        24,
    )
    values = mask.tolist()[0]
    assert 0.0 in values
    assert 1.0 in values
    assert values[-1] == 0.0
