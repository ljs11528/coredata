#!/usr/bin/env python3
"""
Count total pseudo-label pixels generated (all valid pixels in unlabeled set)
and how many pixels our selection method finally chooses (weight==1 in PCOS).

This script replays the selection logic used in training (see
`train/semi_supervised_train.SemiModule.get_weight`) across the unlabeled
dataset using the provided checkpoint and prints totals.

Usage example:
  python count_pseudo_selection.py --config configs/pascal.yaml \
    --ckpt /path/to/ckpt --unlabeled_id_path /path/to/unlabeled.txt \
    --batch_size 8 --max_samples 1000 --device cuda
"""
import argparse
import os
import yaml
import torch
import numpy as np
from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.PCOS import get_max_confidence_and_residual_variance, batch_class_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--unlabeled_id_path', required=False, help='file listing unlabeled image id pairs')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size to simulate training batches')
    parser.add_argument('--max_samples', type=int, default=None, help='limit number of unlabeled samples (for speed)')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


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


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # change cwd to project root like other scripts so relative paths in model code resolve
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
    except Exception:
        pass

    unlabeled_id_path = args.unlabeled_id_path or cfg.get('unlabeled_id_path') or cfg.get('dataset', {}).get('unlabeled_id_path')
    if unlabeled_id_path is None:
        # try conventional location under dataset root
        dataset_root = cfg['dataset']['root']
        unlabeled_id_path = os.path.join(dataset_root, 'train_u_ids.txt')
        if not os.path.exists(unlabeled_id_path):
            raise RuntimeError('Could not find unlabeled id list; provide --unlabeled_id_path')

    # prepare unlabeled dataset (use same crop_size as training to allow batching)
    dataset_root = cfg['dataset']['root']
    crop_size = cfg['dataset'].get('crop_size', None)
    # if max_samples provided, SemiDataset supports nsample argument
    nsample = args.max_samples
    unlabeled = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'train_u', unlabeled_id_path, crop_size=crop_size, nsample=nsample)

    bsize = args.batch_size or cfg.get('batch_size') or 4
    from torch.utils.data import DataLoader
    loader = DataLoader(unlabeled, batch_size=bsize, shuffle=False, num_workers=1, pin_memory=True)

    # build model and load checkpoint
    model = ModelBuilder(cfg['model'])
    load_ckpt_to_model(model, args.ckpt, map_location='cpu')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model.to(device).eval()

    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')

    total_valid_pixels = 0
    total_pseudo_pixels = 0  # number of predicted pseudo-label pixels (valid pixels -> have a pseudo-label)
    total_selected_pixels = 0  # number of pixels our method marks as selected (weight == 1)

    epsilon = 1e-8
    alpha = 2.0

    with torch.no_grad():
        for ib, batch in enumerate(loader):
            # SemiDataset for train_u returns: img_w, img_s, img_m, ignore_mask, cutmix_box, cover_mask
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                imgs = batch[0]
                ignore_masks = batch[3]
            else:
                # fallback: assume batch is (imgs, ...)
                imgs = batch[0]
                ignore_masks = None

            imgs = imgs.to(device)
            outs = model(imgs, False)
            if isinstance(outs, (list, tuple)):
                outs = outs[0]
            probs = outs.softmax(dim=1)

            # build valid mask (ignore_mask != 255) - if ignore_masks not provided, assume all valid
            if ignore_masks is None:
                valid_mask = torch.ones(probs.shape[0], probs.shape[2], probs.shape[3], dtype=torch.bool, device=probs.device)
            else:
                valid_mask = (ignore_masks != 255)

            # total valid pixels in this batch
            total_valid_pixels += int(valid_mask.sum().item())

            # every valid pixel receives a pseudo-label (argmax)
            total_pseudo_pixels += int(valid_mask.sum().item())

            # compute PCOS weights exactly like training
            max_confidence, scaled_residual_variance = get_max_confidence_and_residual_variance(probs.cpu(), valid_mask.cpu(), K, epsilon)
            means, vars = batch_class_stats(max_confidence, scaled_residual_variance, K)

            # move variables to CPU tensors for arithmetic (means/vars are on same device as inputs)
            means = means.to(max_confidence.device)
            vars = vars.to(max_confidence.device)
            max_confidence = max_confidence.to(means.device)
            scaled_residual_variance = scaled_residual_variance.to(means.device)

            conf_mean = means[:, 0].view(-1, 1, 1)
            res_mean = means[:, 1].view(-1, 1, 1)
            conf_var = vars[:, 0].view(-1, 1, 1)
            res_var = vars[:, 1].view(-1, 1, 1)

            conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
            res_z = (res_mean - scaled_residual_variance) / torch.sqrt(res_var + epsilon)

            weight_conf = torch.exp(- (conf_z ** 2) / alpha)
            weight_res = torch.exp(- (res_z ** 2) / alpha)
            weight = weight_conf * weight_res

            confident_mask = (conf_z > 0) | (res_z > 0)
            weight = torch.where(confident_mask, torch.ones_like(weight), weight)

            weight_mask = torch.where(valid_mask.cpu(), weight, torch.zeros_like(weight))

            # selected pixels are those with weight == 1 (as done in training: conf_mask = weight_u_w == 1)
            selected = (weight_mask == 1.0)
            total_selected_pixels += int(selected.sum().item())

            if (ib + 1) % 50 == 0:
                print(f"Processed {ib+1} / {len(loader)} batches. so far selected {total_selected_pixels} / pseudo {total_pseudo_pixels}")

    print('\nSummary:')
    print(f'Total valid pixels (unlabeled dataset): {total_valid_pixels:,}')
    print(f'Total pseudo-label pixels produced (one per valid pixel): {total_pseudo_pixels:,}')
    print(f'Total pixels selected by method (weight==1): {total_selected_pixels:,}')
    frac = (total_selected_pixels / total_pseudo_pixels) if total_pseudo_pixels > 0 else 0.0
    print(f'Fraction selected: {frac:.4f} ({frac*100:.2f}%)')


if __name__ == '__main__':
    main()
