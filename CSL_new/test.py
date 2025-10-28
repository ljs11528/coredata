#!/usr/bin/env python3
"""
Simple evaluation script for a trained .ckpt checkpoint.

Usage (example):
  python test.py --config configs/pascal.yaml --ckpt path/to/checkpoint.ckpt --val_id_path splits/pascal/val.txt --device cuda

The script builds the model from the config, loads the checkpoint weights, runs the validation set and
prints per-class IoU and mean IoU.
"""
import argparse
import yaml
import torch
import os
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
        # common Lightning prefix when the model was stored as self.model
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        # strip 'module.' if saved from DataParallel/Distributed
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        # sometimes checkpoints are nested like 'module.model.'
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        new_state[new_k] = v

    # attempt to load; allow missing/unexpected to be reported
    missing, unexpected = model.load_state_dict(new_state, strict=False) if hasattr(model, 'load_state_dict') else ([], [])
    # PyTorch may return a NamedTuple instead of tuple depending on version
    try:
        if hasattr(missing, 'missing_keys'):
            missing_keys = missing.missing_keys
            unexpected_keys = missing.unexpected_keys
        else:
            missing_keys = missing
            unexpected_keys = unexpected
    except Exception:
        missing_keys = []
        unexpected_keys = []

    if len(missing_keys) > 0:
        print(f"Warning: missing keys when loading checkpoint: {missing_keys[:10]}{'...' if len(missing_keys)>10 else ''}")
    if len(unexpected_keys) > 0:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected_keys[:10]}{'...' if len(unexpected_keys)>10 else ''}")


def center_crop_pred(model, img, crop_size):
    # img: tensor BxCxHxW
    h, w = img.shape[-2:]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    img_c = img[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size]
    with torch.no_grad():
        out = model(img_c.to(next(model.parameters()).device), False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    pred = out.argmax(dim=1).cpu()
    return pred


def sliding_window_pred(model, img, crop_size, nclass):
    b, _, h, w = img.shape
    device = next(model.parameters()).device
    final = torch.zeros(b, nclass, h, w).type_as(img)
    row = 0
    stride = int(crop_size * 2 / 3)
    while row < h:
        col = 0
        while col < w:
            h2 = min(h, row + crop_size)
            w2 = min(w, col + crop_size)
            patch = img[:, :, row:h2, col:w2]
            with torch.no_grad():
                out = model(patch.to(device), False)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # accumulate softmax scores
            final[:, :, row:h2, col:w2] += out.softmax(dim=1).cpu()
            col += stride
        row += stride
    pred = final.argmax(dim=1)
    return pred


def evaluate(cfg, ckpt_path, val_id_path, device='cuda'):
    cfg = dict(cfg)
    model = ModelBuilder(cfg['model'])
    device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    load_ckpt_to_model(model, ckpt_path, map_location='cpu')
    model.to(device).eval()

    valset = SemiDataset(cfg['dataset']['name'], cfg['dataset']['root'], 'val', val_id_path, crop_size=cfg['dataset'].get('crop_size', None))
    from torch.utils.data import DataLoader
    valloader = DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    K = cfg.get('nclass') or cfg.get('train', {}).get('nclass') or cfg['model']['decoder']['kwargs'].get('nclass')
    intersection_meter = torch.zeros(K)
    union_meter = torch.zeros(K)

    eval_mode = cfg.get('train', {}).get('eval_mode', 'original')
    crop_size = cfg.get('crop_size') or cfg.get('dataset', {}).get('crop_size') or cfg.get('train', {}).get('crop_size')

    for idx, batch in enumerate(valloader):
        img, mask, _id = batch
        if eval_mode == 'center_crop' and crop_size is not None:
            pred = center_crop_pred(model, img, crop_size)
        elif eval_mode == 'sliding_window' and crop_size is not None:
            pred = sliding_window_pred(model, img, crop_size, K)
        else:
            with torch.no_grad():
                out = model(img.to(device), False)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred = out.argmax(dim=1).cpu()

        inter, union, target = intersectionAndUnion(pred, mask, K, 255)
        intersection_meter += inter
        union_meter += union

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(valloader)} images")

    iou_class = intersection_meter / (union_meter + 1e-10) * 100.0
    mIOU = iou_class.mean().item()
    print('\nPer-class IoU:')
    for i, iou in enumerate(iou_class):
        print(f"Class {i}: {iou:.2f}%")
    print(f"\nMean IoU: {mIOU:.2f}%")


def build_val_id_from_folders(dataset_root, out_path=None):
    # Look for JPEGImages and Annotations folders and build a val id list where each line is
    # "path/to/image path/to/mask" (paths relative to dataset_root)
    img_dir = os.path.join(dataset_root, 'JPEGImages')
    # Pascal VOC style masks might be in 'Annotations' or 'SegmentationClass' or 'SegmentationObject'
    possible_mask_dirs = ['Annotations', 'SegmentationClass', 'SegmentationObject']
    ann_dir = None
    for d in possible_mask_dirs:
        p = os.path.join(dataset_root, d)
        if os.path.isdir(p):
            ann_dir = p
            ann_dir_name = d
            break
    if ann_dir is None:
        raise FileNotFoundError(f"Couldn't find a mask folder ({possible_mask_dirs}) under {dataset_root}")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Couldn't find JPEGImages or Annotations under {dataset_root}")

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    lines = []
    for img_name in img_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join('JPEGImages', img_name)
        # Ground-truth masks may have different extensions; try png, jpg, or jpeg
        for ext in ('.png', '.jpg', '.jpeg'):
            ann_name = base + ext
            if os.path.exists(os.path.join(ann_dir, ann_name)):
                ann_path = os.path.join(ann_dir_name, ann_name)
                break
        else:
            # if no mask found, skip
            continue
        lines.append(f"{img_path} {ann_path}")

    if out_path is not None:
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        return out_path
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--val_id_path', default=None)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # Ensure relative paths inside model code (like "pretrained/resnet101.pth") resolve
    # by switching working directory to the project root (parent of the config directory).
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
            print(f"Changed working directory to project root: {project_root}")
    except Exception as e:
        print(f"Warning: failed to change working directory to project root: {e}")
    # prefer CLI args; otherwise fall back to config.test
    test_cfg = cfg.get('test', {})
    ckpt = args.ckpt or test_cfg.get('ckpt')
    device = args.device or test_cfg.get('device', 'cpu')
    val_id_path = args.val_id_path or test_cfg.get('val_id_path')

    # if val_id_path is null, try to build one inside the dataset folder and save to a temp file
    if val_id_path is None:
        dataset_root = cfg['dataset']['root']
        tmp_id_path = os.path.join(cfg['dataset']['root'], 'val_generated_ids.txt')
        print(f"Generating val id list at {tmp_id_path} from {dataset_root}...")
        build_val_id_from_folders(dataset_root, out_path=tmp_id_path)
        val_id_path = tmp_id_path

    # make sure eval_mode from config is honored (test > train)
    if 'test' in cfg and 'eval_mode' in cfg['test']:
        cfg.setdefault('train', {})['eval_mode'] = cfg['test']['eval_mode']

    evaluate(cfg, ckpt, val_id_path, device=device)


if __name__ == '__main__':
    main()
