#!/usr/bin/env python3
"""
Compare two trained models on a random subset of the test/val set and save visual results.

Saves for each sampled id:
  - original image (copied from dataset JPEGImages)
  - ground-truth mask (copied from SegmentationClass)
  - model A prediction (PNG of class indices and a colored overlay)
  - model B prediction (PNG of class indices and a colored overlay)

Usage example:
  python compare_models_visualize.py --config CSL_new/configs/pascal.yaml \
      --ckpt_a /workspace/coredata/CSL_new/exp/pascal/CSL/r101/1_4/checkpoints/epoch=66-val_mIOU=78.84.ckpt \
      --ckpt_b /workspace/coredata/CSL/exp/pascal/CSL/r101/1_4/checkpoints/epoch=54-val_mIOU=79.57.ckpt \
      --out_dir /workspace/coredata/CSL_new/compare_results --num_samples 20 --device cuda
"""
import argparse
import os
import random
import shutil
from PIL import Image
import numpy as np
import torch

from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset


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


def save_mask_png(mask_arr, out_path, cmap=None):
    # mask_arr: 2D numpy of class indices
    img = Image.fromarray(mask_arr.astype(np.uint8), mode='L')
    img.save(out_path)


def save_color_overlay(mask_arr, img_arr, out_path, cmap='tab20'):
    # simple colored overlay using matplotlib colormap
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        cmap_obj = cm.get_cmap(cmap)
        colored = cmap_obj(mask_arr % cmap_obj.N)[:, :, :3]  # HxWx3 float
        colored = (colored * 255).astype(np.uint8)
        pil = Image.fromarray(colored)
        pil.save(out_path)
    except Exception:
        # fallback: save grayscale mask
        save_mask_png(mask_arr, out_path)


def infer_model_on_tensor(model, img_tensor, device):
    model.to(device).eval()
    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(device)
        out = model(inp, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1)
        pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt_a', required=True)
    p.add_argument('--ckpt_b', required=True)
    p.add_argument('--val_id_path', required=False, help='optional override of val/test id file')
    p.add_argument('--out_dir', default='compare_results')
    p.add_argument('--num_samples', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    import yaml
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # switch cwd to project root so relative pretrained paths resolve
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
    except Exception:
        pass

    test_cfg = cfg.get('test', {})
    val_id_path = args.val_id_path or test_cfg.get('val_id_path')
    if val_id_path is None:
        dataset_root = cfg['dataset']['root']
        val_id_path = os.path.join(dataset_root, 'val_generated_ids.txt')
        # try to build if helper available
        try:
            from test import build_val_id_from_folders
            build_val_id_from_folders(dataset_root, out_path=val_id_path)
        except Exception:
            pass

    val_lines = read_val_list(val_id_path)
    total = len(val_lines)
    if total == 0:
        raise RuntimeError(f'No lines found in {val_id_path}')

    random.seed(args.seed)
    sample_n = min(args.num_samples, total)
    sampled_idx = sorted(random.sample(range(total), sample_n))

    dataset_root = cfg['dataset']['root']
    # use SemiDataset for transforms so model input is correct
    dataset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', val_id_path, crop_size=cfg['dataset'].get('crop_size', None))

    # load both models
    modelA = ModelBuilder(cfg['model'])
    modelB = ModelBuilder(cfg['model'])
    load_ckpt_to_model(modelA, args.ckpt_a, map_location='cpu')
    load_ckpt_to_model(modelB, args.ckpt_b, map_location='cpu')

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)
    orig_dir = os.path.join(args.out_dir, 'original')
    gt_dir = os.path.join(args.out_dir, 'gt')
    a_dir = os.path.join(args.out_dir, 'modelA')
    b_dir = os.path.join(args.out_dir, 'modelB')
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    for i_idx, idx in enumerate(sampled_idx):
        line = val_lines[idx]
        parts = line.split()
        if len(parts) < 2:
            print(f'skipping malformed line: {line}')
            continue
        img_rel = parts[0]
        mask_rel = parts[1]
        img_path = os.path.join(dataset_root, img_rel)
        mask_path = os.path.join(dataset_root, mask_rel)

        # copy original and GT
        base_name = os.path.splitext(os.path.basename(img_rel))[0]
        out_orig = os.path.join(orig_dir, f'{base_name}.jpg')
        out_gt = os.path.join(gt_dir, f'{base_name}_gt.png')
        try:
            shutil.copyfile(img_path, out_orig)
        except Exception:
            # fallback: open and re-save
            try:
                Image.open(img_path).save(out_orig)
            except Exception as e:
                print(f'warning: cannot copy original {img_path}: {e}')
        try:
            shutil.copyfile(mask_path, out_gt)
        except Exception:
            try:
                Image.open(mask_path).save(out_gt)
            except Exception as e:
                print(f'warning: cannot copy mask {mask_path}: {e}')

        # get preprocessed tensor from dataset
        try:
            img_tensor, mask_tensor, sid = dataset[idx]
        except Exception as e:
            print(f'error accessing dataset[{idx}]: {e}')
            continue

        # run both models
        predA = infer_model_on_tensor(modelA, img_tensor, device)
        predB = infer_model_on_tensor(modelB, img_tensor, device)

        # save predictions as PNG (class indices) and colored overlay
        out_a_mask = os.path.join(a_dir, f'{base_name}_predA.png')
        out_b_mask = os.path.join(b_dir, f'{base_name}_predB.png')
        save_mask_png(predA, out_a_mask)
        save_mask_png(predB, out_b_mask)

        # also save colored overlays (best-effort)
        try:
            orig_img = np.array(Image.open(img_path).convert('RGB'))
            save_color_overlay(predA, orig_img, os.path.join(a_dir, f'{base_name}_predA_color.png'))
            save_color_overlay(predB, orig_img, os.path.join(b_dir, f'{base_name}_predB_color.png'))
        except Exception:
            pass

        print(f'[{i_idx+1}/{sample_n}] saved: {base_name}')


if __name__ == '__main__':
    main()
