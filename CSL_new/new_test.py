#!/usr/bin/env python3
"""
Compute per-image accuracy and mean max-confidence, save CSV and plot scatter.

Each validation sample produces:
 - accuracy (%) on labelled pixels (ignore_index=255)
 - mean of maximum softmax confidence across valid pixels (0..1)

Plot: x-axis = accuracy (%), y-axis = mean max-confidence. Add horizontal line at y=0.95 and vertical line at x=75.

Usage:
  python new_test.py --config configs/pascal.yaml --ckpt path/to.ckpt --val_id_path splits/pascal/val.txt --out_dir results
"""
import argparse
import yaml
import torch
import os
import csv
import matplotlib.pyplot as plt
from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.utils import intersectionAndUnion


def load_ckpt_to_model(model, ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)


def compute_per_image_stats(model, dataloader, device, K):
    model.to(device).eval()
    results = []  # list of dicts: {'id': id, 'accuracy': float, 'mean_max_conf': float}

    for idx, batch in enumerate(dataloader):
        img, mask, id = batch
        img = img.to(device)
        with torch.no_grad():
            out = model(img, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1).cpu()  # B x C x H x W
        pred = probs.argmax(dim=1)  # B x H x W

        # compute per-sample accuracy and mean max-confidence
        for b in range(pred.shape[0]):
            p = pred[b]
            m = mask[b]
            valid = (m != 255)
            if valid.sum() == 0:
                acc = float('nan')
                mean_max_conf = float('nan')
            else:
                correct = (p[valid] == m[valid]).sum().item()
                acc = 100.0 * correct / int(valid.sum())
                max_conf = probs[b].max(dim=0)[0]  # H x W
                mean_max_conf = float(max_conf[valid].mean().item())

            results.append({'id': id[b], 'accuracy': acc, 'mean_max_conf': mean_max_conf})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1} / {len(dataloader)} batches")

    return results


def plot_results(results, out_dir):
    xs = [r['accuracy'] for r in results if not (r['accuracy'] is None or (isinstance(r['accuracy'], float) and (torch.isnan(torch.tensor(r['accuracy'])).item())))]
    ys = [r['mean_max_conf'] for r in results if not (r['mean_max_conf'] is None or (isinstance(r['mean_max_conf'], float) and (torch.isnan(torch.tensor(r['mean_max_conf'])).item())))]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=10, alpha=0.6)
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Mean Max-Confidence')
    plt.title('Per-sample Accuracy vs Mean Max-Confidence')
    # horizontal line at y=0.95
    plt.axhline(0.95, color='red', linestyle='--', linewidth=1)
    # vertical line at x=75 (%)
    plt.axvline(75.0, color='blue', linestyle='--', linewidth=1)
    plt.grid(True, linestyle=':', linewidth=0.5)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'accuracy_vs_confidence.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved plot to {fig_path}")


def save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'per_image_confidence_accuracy.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'accuracy', 'mean_max_conf'])
        writer.writeheader()
        for r in results:
            # write id as str
            rid = r['id'] if isinstance(r['id'], str) else r['id'][0] if isinstance(r['id'], (list, tuple)) else str(r['id'])
            writer.writerow({'id': rid, 'accuracy': r['accuracy'], 'mean_max_conf': r['mean_max_conf']})
    print(f"Saved CSV to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=False)
    parser.add_argument('--val_id_path', required=False)
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--max_samples', type=int, default=None, help='If set, randomly sample up to this many val samples (deterministic seed=42)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # change cwd to project root as in test.py
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
            print(f"Changed working directory to project root: {project_root}")
    except Exception:
        pass

    test_cfg = cfg.get('test', {})
    ckpt = args.ckpt or test_cfg.get('ckpt')
    val_id_path = args.val_id_path or test_cfg.get('val_id_path')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    if val_id_path is None:
        dataset_root = cfg['dataset']['root']
        val_id_path = os.path.join(dataset_root, 'val_generated_ids.txt')
        from test import build_val_id_from_folders
        build_val_id_from_folders(dataset_root, out_path=val_id_path)

    # If requested, sample a deterministic subset of val ids
    if args.max_samples is not None:
        import random
        random.seed(42)
        with open(val_id_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) == 0:
            raise RuntimeError(f'val_id_path {val_id_path} contains no entries')
        sample_n = min(len(lines), args.max_samples)
        sampled = random.sample(lines, sample_n)
        sampled_path = os.path.join(args.out_dir, f'val_sample_{sample_n}.txt')
        os.makedirs(args.out_dir, exist_ok=True)
        with open(sampled_path, 'w') as f:
            f.write('\n'.join(sampled))
        print(f'Using sampled val list {sampled_path} ({sample_n} entries)')
        val_id_path = sampled_path

    # load model
    model = ModelBuilder(cfg['model'])
    load_ckpt_to_model(model, ckpt, map_location='cpu')

    # build dataloader
    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', val_id_path, crop_size=cfg['dataset'].get('crop_size', None))
    from torch.utils.data import DataLoader
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')
    results = compute_per_image_stats(model, valloader, device, K)

    save_csv(results, args.out_dir)
    plot_results(results, args.out_dir)


if __name__ == '__main__':
    main()
