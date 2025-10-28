#!/usr/bin/env python3
"""
Evaluate model on a subset of validation samples, split classes into majority/minority
based on class pixel-frequency across the selected samples, then plot per-sample
accuracy vs mean max-confidence colored by whether the sample's dominant class
is majority (red) or minority (blue).

Assumptions:
- The class frequency split is computed from the mask pixels of the samples being evaluated
  (i.e., the sampled val list). We split classes into two groups by sorting total pixel
  counts and assigning the top 50% classes as majority, bottom 50% as minority.
- Each sample is assigned to majority/minority based on its dominant class (the class
  with the largest pixel count in that sample's ground-truth mask).

Usage:
  python new_test_majority.py --config configs/pascal.yaml --ckpt path/to.ckpt --out_dir results --max_samples 128
"""
import argparse
import yaml
import torch
import os
import csv
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.utils import intersectionAndUnion
from util.classes import CLASSES


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


def read_val_list(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def build_class_pixel_counts(val_lines, dataset_root, K):
    """
    Count pixels per class across the provided val_lines.
    val_lines: list of strings like 'JPEGImages/2007_000033.jpg SegmentationClass/2007_000033.png'
    Returns: numpy array shape (K,) with counts (int)
    """
    counts = np.zeros(K, dtype=np.int64)
    for line in val_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        mask_rel = parts[1]
        mask_path = os.path.join(dataset_root, mask_rel)
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.int64)
        mask = mask[mask != 255]
        if mask.size == 0:
            continue
        # clip to [0, K-1]
        mask = mask[(mask >= 0) & (mask < K)]
        if mask.size == 0:
            continue
        hist = np.bincount(mask, minlength=K)
        counts += hist
    return counts


def determine_majority_classes(counts):
    # sort classes by counts descending and pick the top 40% as majority (user request)
    K = counts.shape[0]
    order = np.argsort(-counts)
    top_n = int(np.ceil(K * 0.4))
    if top_n < 1:
        top_n = 1
    top_half = order[:top_n]
    majority_mask = np.zeros(K, dtype=bool)
    majority_mask[top_half] = True
    return majority_mask


def compute_per_image_stats_and_groups(model, dataloader, device, K, majority_mask, select_top_k=None):
    # This function now supports batch-wise pseudo-label selection simulation.
    model.to(device).eval()
    results = []
    for idx, batch in enumerate(dataloader):
        img, mask, id = batch
        # img shape: B x C x H x W; we will treat this B as the batch for selection
        bsize = img.shape[0]
        img = img.to(device)
        with torch.no_grad():
            out = model(img, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1).cpu()
        pred = probs.argmax(dim=1)

        # collect per-sample stats for this mini-batch
        batch_stats = []
        for b in range(pred.shape[0]):
            p = pred[b]
            # move prediction to numpy for per-pixel comparison
            p_np = p.cpu().numpy()
            m = mask[b].numpy()
            valid_mask = (m != 255)
            if valid_mask.sum() == 0:
                miou = float('nan')
                mean_max_conf = float('nan')
                dominant = -1
                group = 'unknown'
            else:
                # compute per-class intersection and union for this sample
                ious = []
                for c in range(K):
                    inter = int(((p_np == c) & (m == c) & valid_mask).sum())
                    union = int((((p_np == c) | (m == c)) & valid_mask).sum())
                    if union > 0:
                        ious.append(inter / union)
                if len(ious) == 0:
                    miou = float('nan')
                else:
                    miou = 100.0 * float(np.mean(ious))

                max_conf = probs[b].max(dim=0)[0]
                mean_max_conf = float(max_conf[valid_mask].mean().item())

                # determine dominant class in this mask
                vals, counts = np.unique(m[valid_mask], return_counts=True)
                valid_idxs = (vals >= 0) & (vals < K)
                if valid_idxs.sum() == 0:
                    dominant = -1
                    group = 'unknown'
                else:
                    vals = vals[valid_idxs]
                    counts = counts[valid_idxs]
                    dominant = int(vals[np.argmax(counts)])
                    group = 'majority' if majority_mask[dominant] else 'minority'

            rid = id[b] if isinstance(id[b], str) else id[b][0] if isinstance(id[b], (list, tuple)) else str(id[b])
            batch_stats.append({'id': rid, 'miou': miou, 'mean_max_conf': mean_max_conf, 'dominant_class': dominant, 'group': group})

        # If selection is requested, pick top-k samples in this mini-batch by mean_max_conf
        if select_top_k is not None and len(batch_stats) > 0:
            # build array of confs and mask NaNs
            confs = np.array([s['mean_max_conf'] for s in batch_stats], dtype=np.float64)
            valid_mask = ~np.isnan(confs)
            k = int(min(select_top_k, int(valid_mask.sum())))
            # default none selected
            for s in batch_stats:
                s['selected'] = False
            if k > 0 and valid_mask.sum() > 0:
                # choose top-k indices among valid ones
                valid_indices = np.where(valid_mask)[0]
                sorted_idx = valid_indices[np.argsort(-confs[valid_indices])]
                chosen = sorted_idx[:k]
                for idx_ch in chosen:
                    batch_stats[idx_ch]['selected'] = True
        else:
            for s in batch_stats:
                s['selected'] = False

        results.extend(batch_stats)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1} / {len(dataloader)} batches")

    return results


def plot_colored(results, out_dir, class_names=None):
    xs_major = [r.get('miou', float('nan')) for r in results if r['group'] == 'majority']
    ys_major = [r['mean_max_conf'] for r in results if r['group'] == 'majority']
    xs_minor = [r.get('miou', float('nan')) for r in results if r['group'] == 'minority']
    ys_minor = [r['mean_max_conf'] for r in results if r['group'] == 'minority']

    plt.figure(figsize=(8, 6))
    # minority blue, majority red
    plt.scatter(xs_minor, ys_minor, s=18, alpha=0.7, c='blue', label='Minority Class')
    plt.scatter(xs_major, ys_major, s=18, alpha=0.7, c='red', label='Majority Class')
    plt.xlabel('mIoU (%)')
    plt.ylabel('Max-Confidence')
    plt.title('Per-sample mIoU vs Max-Confidence (majority=red, minority=blue)')
    # horizontal threshold line at mean-max-confidence = 0.95
    plt.axhline(0.95, color='black', linestyle='--', linewidth=1, label='conf=0.95')

    # compute and draw vertical lines for requested statistics
    # mean accuracy of samples with mean_max_conf > 0.95
    miou_all = np.array([r.get('miou', float('nan')) for r in results], dtype=np.float64)
    confs = np.array([r['mean_max_conf'] for r in results], dtype=np.float64)
    # mask NaNs
    valid_miou_mask = ~np.isnan(miou_all)
    valid_conf_mask = ~np.isnan(confs)
    # overall mean mIoU (exclude NaNs)
    if valid_miou_mask.sum() > 0:
        mean_miou_all = float(np.nanmean(miou_all[valid_miou_mask]))
        plt.axvline(mean_miou_all, color='orange', linestyle='--', linewidth=1, label=f'mean mIoU (all)={mean_miou_all:.2f}%')
    else:
        mean_miou_all = None

    # mean mIoU for samples with mean_max_conf > 0.95
    high_conf_mask = (confs > 0.95) & valid_miou_mask & valid_conf_mask
    if high_conf_mask.sum() > 0:
        mean_miou_conf95 = float(np.nanmean(miou_all[high_conf_mask]))
        plt.axvline(mean_miou_conf95, color='green', linestyle='--', linewidth=1, label=f'mean mIoU (conf>0.95)={mean_miou_conf95:.2f}%')
    else:
        mean_miou_conf95 = None
    # if selection information is present, overlay selected samples with a black circle outline
    selected_flags = np.array([bool(r.get('selected', False)) for r in results])
    if selected_flags.sum() > 0:
        xs_all = np.array([r.get('miou', float('nan')) for r in results], dtype=np.float64)
        ys_all = np.array([r['mean_max_conf'] for r in results], dtype=np.float64)
        sel_mask = selected_flags & (~np.isnan(xs_all)) & (~np.isnan(ys_all))
        # additionally, highlight minority-class samples with high mIoU (>0.85)
        # handle threshold unit: if miou values appear to be percentages (>1.5), treat 0.85 as 85
        try:
            thr_frac = 0.85
            if np.nanmax(xs_all) > 1.5:
                thr = thr_frac * 100.0
            else:
                thr = thr_frac
        except Exception:
            thr = 0.85

        minority_mask = np.array([1 if r.get('group') == 'minority' else 0 for r in results], dtype=bool)
        minority_high = minority_mask & (~np.isnan(xs_all)) & (xs_all >= thr)

        # combined mask: selected by method OR minority with high mIoU
        combined_mask = sel_mask | minority_high

        # plot combined highlighted samples as black circle outlines on top of the colored points
        if combined_mask.sum() > 0:
            plt.scatter(xs_all[combined_mask], ys_all[combined_mask], s=48, facecolors='none', edgecolors='k', linewidths=0.9, label='selected / minority (miou>0.85)')

        # draw a vertical line at the mean mIoU of method-selected pseudo-labels (selected_flags only)
        try:
            sel_mious = xs_all[sel_mask]
            if sel_mious.size > 0:
                mean_sel_miou = float(np.nanmean(sel_mious))
                plt.axvline(mean_sel_miou, color='black', linestyle='-', linewidth=1, label=f'mean mIoU (selected)={mean_sel_miou:.2f}%')
        except Exception:
            pass
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'accuracy_vs_confidence_majority.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved colored plot to {fig_path}")


def save_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'per_image_confidence_accuracy_majority.csv')
    with open(csv_path, 'w', newline='') as f:
        # include 'selected' flag if present
        fieldnames = ['id', 'miou', 'mean_max_conf', 'dominant_class', 'group']
        if any('selected' in r for r in results):
            fieldnames.append('selected')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # ensure we write miou field (some results may still contain 'accuracy' if older runs exist)
            out_r = r.copy()
            if 'miou' not in out_r and 'accuracy' in out_r:
                out_r['miou'] = out_r.pop('accuracy')
            writer.writerow(out_r)
    print(f"Saved CSV to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=False)
    parser.add_argument('--val_id_path', required=False)
    parser.add_argument('--out_dir', default='results_majority')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--selection_batch_size', type=int, default=None, help='Batch size to simulate selection (m)')
    parser.add_argument('--select_top_k', type=int, default=None, help='Select top k samples per batch as pseudo-labels')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # switch cwd like other scripts
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
    if val_id_path is None:
        dataset_root = cfg['dataset']['root']
        val_id_path = os.path.join(dataset_root, 'val_generated_ids.txt')
        from test import build_val_id_from_folders
        build_val_id_from_folders(dataset_root, out_path=val_id_path)

    # if sampling requested, sample lines and write sampled file
    if args.max_samples is not None:
        random.seed(42)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(val_id_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        sample_n = min(len(lines), args.max_samples)
        sampled = random.sample(lines, sample_n)
        sampled_path = os.path.join(args.out_dir, f'val_sample_{sample_n}.txt')
        with open(sampled_path, 'w') as f:
            f.write('\n'.join(sampled))
        print(f'Using sampled val list {sampled_path} ({sample_n} entries)')
        val_lines = sampled
        val_id_path = sampled_path
    else:
        val_lines = read_val_list(val_id_path)

    dataset_root = cfg['dataset']['root']
    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')

    # compute class pixel counts across chosen val lines
    print('Counting class pixels across selected validation samples...')
    counts = build_class_pixel_counts(val_lines, dataset_root, K)
    majority_mask = determine_majority_classes(counts)
    print('Class counts:', counts)
    print('Majority classes (indices):', np.where(majority_mask)[0].tolist())

    # build dataloader for sampled val. Use batch_size=1 to avoid collate issues with variable image sizes.
    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', val_id_path, crop_size=cfg['dataset'].get('crop_size', None))
    from torch.utils.data import DataLoader
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    # load model and run
    model = ModelBuilder(cfg['model'])
    load_ckpt_to_model(model, ckpt, map_location='cpu')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    # collect per-sample stats (batch_size=1 loader)
    results = compute_per_image_stats_and_groups(model, valloader, device, K, majority_mask)

    # If selection parameters provided, simulate per-batch selection by grouping the results into
    # chunks of size selection_batch_size and selecting top-k by mean_max_conf within each chunk.
    if args.selection_batch_size is not None and args.select_top_k is not None:
        m = int(args.selection_batch_size)
        k = int(args.select_top_k)
        # initialize selected flags
        for r in results:
            r['selected'] = False
        for start in range(0, len(results), m):
            chunk = results[start: start + m]
            confs = np.array([c['mean_max_conf'] for c in chunk], dtype=np.float64)
            valid_mask = ~np.isnan(confs)
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size == 0:
                continue
            kk = min(k, valid_indices.size)
            sorted_idx = valid_indices[np.argsort(-confs[valid_indices])]
            chosen = sorted_idx[:kk]
            for ci in chosen:
                chunk[ci]['selected'] = True

    save_csv(results, args.out_dir)
    plot_colored(results, args.out_dir, class_names=CLASSES.get(cfg['dataset']['name'], None))


if __name__ == '__main__':
    main()
