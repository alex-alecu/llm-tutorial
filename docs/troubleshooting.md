# Troubleshooting

## Start here

```bash
source .venv/bin/activate
macllm doctor
python -m pytest
```

## The process was killed or macOS became unresponsive

Run only one ML process. Close memory-heavy applications and use a smaller preset or batch:

```bash
macllm train --preset quick
```

For Qwen, keep `--batch-size 1`, `--grad-checkpoint`, and a 4-bit base model.

## Hugging Face download failed

Check network access and free disk. Downloads are cached under `~/.cache/huggingface`. Re-running normally resumes. Gated models require accepting their terms and `huggingface-cli login`; the models used in this tutorial are public.

## Output directory already exists

Runs never silently mix or overwrite. Choose another output name. Remove an old run only after deciding its checkpoints and metrics are disposable.

## Loss is `nan`

Try a lower learning rate for fine-tuning. For pretraining, first reproduce with `quick`; then inspect your local corpus for empty, binary, or extremely repetitive content.

## Generation is nonsense

- A 20-step smoke checkpoint should be nonsense.
- A TinyStories model only learns small English stories.
- Use a prompt shaped like its training data.
- Check that validation loss fell substantially.
- Fine-tuning cannot install broad knowledge that never existed in the base model.
