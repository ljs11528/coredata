Evaluation (test.py)
=====================

This project provides a simple evaluation script `test.py` to run a trained PyTorch Lightning checkpoint (`.ckpt`) on the validation split and compute per-class IoU and mean IoU.

Basic usage:

1. Prepare a config (for example `configs/pascal.yaml`).
2. Prepare a validation id file (example path in repository: `splits/pascal/val.txt`). Each line should contain: `path/to/image path/to/mask` relative to the dataset root configured in the YAML.
3. Run:

```bash
python test.py --config configs/pascal.yaml --ckpt /path/to/checkpoint.ckpt --val_id_path splits/pascal/val.txt --device cuda
```

Notes:
- `test.py` attempts to be compatible with checkpoints saved from Lightning (it strips common prefixes like `model.` and `module.` when loading weights).
- If your checkpoint contains a different layout you may need to adapt `load_ckpt_to_model` in `test.py`.
