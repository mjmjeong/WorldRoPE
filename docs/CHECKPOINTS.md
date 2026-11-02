# Checkpoint families

Inference-only artifacts are grouped as below, with a portable `config.yaml` beside each `model.pt`. They preserve all model tensors and the training step; optimizer and scheduler state are omitted. Original training checkpoints remain unchanged. WorldRoPE and baseline weights with their model card are published on [Hugging Face](https://huggingface.co/mijeongkim/WorldRoPE). The repository includes all four checkpoint/configuration pairs listed below.

| Family | Dataset | Parameters | Source |
| --- | --- | ---: | --- |
| `rayrope_lvsm_reproduce` | RE10K | 47,365,068 | Official RayRoPE small pretrained checkpoint |
| `urope_lvsm_reproduce` | RE10K | 24,489,024 | Existing vanilla URoPE clean-view step-80,000 checkpoint |
| `worldrope_lvsm` | RE10K | 24,544,392 | Preserved paper main checkpoint |
| `worldrope_lvsm` | Objaverse | 24,544,392 | Preserved paper main checkpoint |

The family suffix `reproduce` is an artifact label; no fresh baseline training was performed during release preparation. Each saved checkpoint has been strict-loaded, and its companion configuration records the model architecture. RayRoPE uses the original 47M architecture; it must not be loaded with the launcher's default 25M integration recipe.

```bash
python scripts/check_checkpoint.py --checkpoint /path/to/checkpoints/rayrope_lvsm_reproduce/re10k/model.pt --config configs/rayrope_official_re10k.yaml
bash scripts/nvs.sh --dataset re10k --method rayrope \
  --model-config configs/rayrope_official_re10k.yaml --gpus 0 \
  --data-root /path/to/re10k --test \
  --checkpoint /path/to/checkpoints/rayrope_lvsm_reproduce/re10k/model.pt \
  --output-dir /path/to/new-evaluation-output --dry-run
```

Use `configs/urope_reproduce_re10k.yaml` with `--method urope` for the URoPE artifact. Use the same checker and scripts with the Objaverse model configuration. `--model-config` supplies the model architecture only; the launcher's other arguments remain explicit.

## Download checkpoints

```bash
hf download mijeongkim/WorldRoPE --local-dir checkpoints
python scripts/check_checkpoint.py \
  --checkpoint checkpoints/worldrope_lvsm/re10k/model.pt \
  --config checkpoints/worldrope_lvsm/re10k/config.yaml
```

Replace `re10k` with `objaverse` for the corresponding model. The Objaverse
checkpoint must be evaluated with `--dataset objaverse` and its packed data;
the included evaluator uses gap 4, whereas the paper reports mixed-gap results.
