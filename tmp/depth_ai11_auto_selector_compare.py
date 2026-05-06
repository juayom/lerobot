from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from controlnet_aux.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from controlnet_aux.segment_anything.predictor import SamPredictor

from lerobot.datasets.genaug_rgbd_masks import build_rgbd_object_masks
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.rgbd_object_aug import predict_masks_for_method
from lerobot.genaug.geometry.depth_utils import sanitize_depth
from lerobot.utils.mask_debug_utils import overlay_mask, save_image

REPO_ID = 'yoohoolala/depth_ai11'
IMAGE_KEY = 'observation.images.intel'
DEPTH_KEY = 'observation.images.intel_depth'
FRAMES = [4, 23, 33, 80, 140, 200]
OUT_ROOT = Path('outputs/depth_ai11_auto_selector_comparison_20260506')
METHODS = ['seeded_grabcut', 'sam_prompt_box', 'sam_auto_selector', 'sam_auto_selector_depth_gated']
AUTO_SCALE = 0.5
SAM_CHECKPOINT = "/home/capstone/jua/lerobot/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = 'vit_h'


def to_np(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == 'f' and float(arr.max(initial=0.0)) <= 1.5:
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def centroid_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def ensure_binary(mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, valid_mask)


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    area = int((mask > 0).sum())
    out = {'area': area}
    bbox = bbox_from_mask(mask)
    cen = centroid_from_mask(mask)
    if bbox is not None:
        x, y, w, h = bbox
        out['bbox'] = bbox
        out['centroid'] = cen
        out['width_ratio'] = float(w / mask.shape[1])
        out['height_ratio'] = float(h / mask.shape[0])
        out['aspect_h_over_w'] = float(h / max(w, 1))
        out['rectangularity'] = float(area / max(w * h, 1))
    return out


def localization_pass(fg_mask: np.ndarray) -> tuple[bool, str]:
    stats = mask_stats(fg_mask)
    area_ratio = stats['area'] / float(fg_mask.shape[0] * fg_mask.shape[1])
    bbox = stats.get('bbox')
    if bbox is None:
        return False, 'foreground empty'
    width_ratio = stats.get('width_ratio', 0.0)
    height_ratio = stats.get('height_ratio', 0.0)
    if area_ratio < 0.004:
        return False, 'foreground too small'
    if area_ratio > 0.22:
        return False, 'foreground too broad'
    if width_ratio > 0.55 or height_ratio > 0.55:
        return False, 'foreground bbox too broad'
    return True, 'foreground localized in a plausible object region'


def background_edit_pass(fg_mask: np.ndarray) -> tuple[bool, str]:
    area_ratio = (fg_mask > 0).sum() / float(fg_mask.shape[0] * fg_mask.shape[1])
    bg_ratio = 1.0 - area_ratio
    if bg_ratio < 0.55:
        return False, 'too little background left for safe edit'
    return True, 'ample background remains outside foreground'


def sam_model_bundle():
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.eval()
    predictor = SamPredictor(sam)
    auto = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,
    )
    return predictor, auto


def best_sam_mask(predictor: SamPredictor, rgb: np.ndarray, box: list[int] | None) -> np.ndarray:
    predictor.set_image(rgb)
    kwargs: dict[str, Any] = {'multimask_output': True}
    if box is not None:
        x, y, w, h = box
        kwargs['box'] = np.array([x, y, x + w - 1, y + h - 1], dtype=np.float32)
    masks, scores, _ = predictor.predict(**kwargs)
    best = masks[int(np.argmax(scores))]
    return (best.astype(np.uint8) * 255)


def rgb_mean(rgb: np.ndarray, mask: np.ndarray) -> list[float]:
    sel = mask > 0
    if not np.any(sel):
        return [0.0, 0.0, 0.0]
    vals = rgb[sel].reshape(-1, 3).astype(np.float32)
    return [float(vals[:, 0].mean()), float(vals[:, 1].mean()), float(vals[:, 2].mean())]


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mask_from_segmentation(seg: Any, valid: np.ndarray) -> np.ndarray:
    return ensure_binary(np.asarray(seg).astype(np.uint8) * 255, valid)


def upscale_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return (cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255


def border_penalty(mask: np.ndarray, border: int = 12) -> float:
    if mask.size == 0:
        return 0.0
    m = mask > 0
    if not np.any(m):
        return 0.0
    edge = np.zeros_like(m, dtype=bool)
    edge[:border, :] = True
    edge[-border:, :] = True
    edge[:, :border] = True
    edge[:, -border:] = True
    return float(np.logical_and(m, edge).sum() / max(m.sum(), 1))


def region_scores(mask: np.ndarray) -> tuple[float, float]:
    cen = centroid_from_mask(mask)
    if cen is None:
        return 0.0, 0.0
    h, w = mask.shape
    cx, cy = cen
    bottle_region = max(0.0, 1.0 - ((abs(cx - w * 0.78) / (w * 0.32)) + (abs(cy - h * 0.30) / (h * 0.28))) / 2.0)
    box_region = max(0.0, 1.0 - ((abs(cx - w * 0.66) / (w * 0.34)) + (abs(cy - h * 0.62) / (h * 0.24))) / 2.0)
    return float(min(bottle_region, 1.0)), float(min(box_region, 1.0))


def depth_consistency(mask: np.ndarray, table_removed: np.ndarray, band_mask: np.ndarray) -> float:
    return 0.55 * iou(mask, table_removed) + 0.45 * iou(mask, band_mask)


def rectangularity(mask: np.ndarray) -> float:
    stats = mask_stats(mask)
    return float(stats.get('rectangularity', 0.0))


def width_height(mask: np.ndarray) -> tuple[int, int]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0, 0
    return int(bbox[2]), int(bbox[3])


def score_auto_candidates(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, auto_masks: list[dict[str, Any]], seeded: dict[str, np.ndarray], source_diag: dict[str, Any], depth_gated: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    h, w = valid.shape
    table_removed = seeded['foreground']
    bottle_seed = seeded['bottle']
    box_seed = seeded['box']
    fg_bbox = bbox_from_mask(table_removed) or [0, 0, w, h]
    x, y, bw, bh = fg_bbox
    roi_mask = np.zeros_like(valid)
    roi_mask[max(0, y - 20):min(h, y + bh + 20), max(0, x - 20):min(w, x + bw + 20)] = 255
    band_mask = cv2.dilate(table_removed, np.ones((15, 15), np.uint8), iterations=1)

    bottle_rows = []
    box_rows = []
    filtered = []
    all_masks_overlay = rgb.copy()

    for idx, raw in enumerate(auto_masks):
        mask = mask_from_segmentation(raw['segmentation'], valid)
        mask = cv2.bitwise_and(mask, roi_mask)
        area = int((mask > 0).sum())
        if area < 900:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        mw, mh = bbox[2], bbox[3]
        area_ratio = area / float(h * w)
        if area_ratio > 0.18:
            continue
        bp = border_penalty(mask)
        bottle_region, box_region = region_scores(mask)
        aspect = mh / max(mw, 1)
        rect = rectangularity(mask)
        mean_rgb = rgb_mean(rgb, mask)
        yellow = np.array([215.0, 190.0, 90.0], dtype=np.float32)
        color_dist = float(np.linalg.norm(np.array(mean_rgb, dtype=np.float32) - yellow))
        color_score = max(0.0, 1.0 - color_dist / 170.0)
        depth_score = depth_consistency(mask, table_removed, band_mask)
        seed_bottle_iou = iou(mask, bottle_seed)
        seed_box_iou = iou(mask, box_seed)
        right_bias = max(0.0, 1.0 - abs((bbox[0] + bbox[2] * 0.5) - w * 0.78) / (w * 0.30))
        upper_bias = max(0.0, 1.0 - abs((bbox[1] + bbox[3] * 0.5) - h * 0.30) / (h * 0.28))
        lower_bias = max(0.0, 1.0 - abs((bbox[1] + bbox[3] * 0.5) - h * 0.62) / (h * 0.25))
        small_pref = max(0.0, 1.0 - area_ratio / 0.06)
        large_pref = min(area_ratio / 0.08, 1.0)
        narrow_pref = max(0.0, 1.0 - min(mw / (w * 0.12), 1.0))
        bottle_score = (
            2.0 * bottle_region + 1.6 * small_pref + 1.8 * min(aspect / 2.4, 1.2) + 1.2 * narrow_pref +
            1.0 * right_bias + 0.8 * upper_bias + 1.1 * depth_score + 0.7 * seed_bottle_iou -
            1.1 * seed_box_iou - 1.6 * bp - 0.9 * max(0.0, area_ratio - 0.08) * 10.0
        )
        box_score = (
            1.9 * box_region + 1.8 * large_pref + 1.4 * min((mw / max(mh, 1)) / 2.2, 1.1) + 1.0 * rect +
            1.1 * lower_bias + 1.1 * color_score + 1.0 * depth_score + 0.8 * seed_box_iou -
            0.8 * seed_bottle_iou - 1.4 * bp
        )
        if depth_gated:
            bottle_score += 0.9 * depth_score
            box_score += 0.9 * depth_score
            if depth_score < 0.18:
                bottle_score -= 2.5
                box_score -= 2.5
        filtered.append({
            'idx': idx,
            'mask': mask,
            'bbox_xywh': bbox,
            'area': area,
            'aspect_ratio': float(aspect),
            'rectangularity': float(rect),
            'bottle_region_score': float(bottle_region),
            'box_region_score': float(box_region),
            'depth_consistency_score': float(depth_score),
            'rgb_mean': mean_rgb,
            'color_score': float(color_score),
            'bottle_score': float(bottle_score),
            'box_score': float(box_score),
            'border_penalty': float(bp),
            'seed_bottle_iou': float(seed_bottle_iou),
            'seed_box_iou': float(seed_box_iou),
        })
        color = ((37 * idx) % 255, (91 * idx) % 255, (173 * idx) % 255)
        all_masks_overlay = overlay_mask(all_masks_overlay, mask, color)

    filtered.sort(key=lambda r: r['area'], reverse=True)
    bottle_rows = sorted(filtered, key=lambda r: r['bottle_score'], reverse=True)
    selected_bottle = bottle_rows[0] if bottle_rows else None

    def pair_box_score(row: dict[str, Any], bottle_row: dict[str, Any] | None) -> float:
        score = float(row['box_score'])
        if bottle_row is not None:
            bcen = centroid_from_mask(bottle_row['mask'])
            xcen = centroid_from_mask(row['mask'])
            if bcen is not None and xcen is not None:
                score += 0.8 if xcen[1] > bcen[1] else -0.8
                score += 0.6 * max(0.0, 1.0 - abs(xcen[0] - bcen[0]) / (w * 0.22))
            score -= 0.7 * iou(row['mask'], bottle_row['mask'])
        return score

    box_rows = sorted(filtered, key=lambda r: pair_box_score(r, selected_bottle), reverse=True)
    selected_box = None
    for row in box_rows:
        if selected_bottle is None or row['idx'] != selected_bottle['idx']:
            selected_box = dict(row)
            selected_box['paired_box_score'] = pair_box_score(row, selected_bottle)
            break
    if selected_bottle is not None:
        selected_bottle = dict(selected_bottle)
    if selected_box is None and box_rows:
        selected_box = dict(box_rows[0])
        selected_box['paired_box_score'] = pair_box_score(box_rows[0], selected_bottle)

    bottle_mask = np.zeros_like(valid)
    box_mask = np.zeros_like(valid)
    failure_reason = ''
    if selected_bottle is None or selected_box is None:
        failure_reason = 'insufficient_auto_candidates'
    elif selected_bottle['idx'] == selected_box['idx']:
        failure_reason = 'same_mask_selected_for_both'
    else:
        bottle_mask = selected_bottle['mask']
        box_mask = cv2.subtract(selected_box['mask'], cv2.dilate(bottle_mask, np.ones((5, 5), np.uint8), iterations=1))
        if int((box_mask > 0).sum()) < 900:
            failure_reason = 'box_removed_by_overlap'
            box_mask = np.zeros_like(valid)

    fg_mask = cv2.bitwise_or(bottle_mask, box_mask)
    bg_mask = cv2.bitwise_and(cv2.bitwise_not(fg_mask), valid)
    sem_verified = bool(source_diag.get('semantic_label_verified', False))
    if not sem_verified and not failure_reason:
        failure_reason = source_diag.get('failure_reason') or 'semantic_label_verified=false'

    diag = {
        'num_auto_masks': len(auto_masks),
        'candidate_count_after_filter': len(filtered),
        'bottle_candidate_score': float(selected_bottle['bottle_score']) if selected_bottle else 0.0,
        'box_candidate_score': float(selected_box.get('paired_box_score', selected_box['box_score'])) if selected_box else 0.0,
        'bottle_bbox_xywh': selected_bottle['bbox_xywh'] if selected_bottle else None,
        'box_bbox_xywh': selected_box['bbox_xywh'] if selected_box else None,
        'bottle_area': int((bottle_mask > 0).sum()),
        'box_area': int((box_mask > 0).sum()),
        'bottle_aspect_ratio': float(selected_bottle['aspect_ratio']) if selected_bottle else 0.0,
        'box_rectangularity': float(selected_box['rectangularity']) if selected_box else 0.0,
        'bottle_region_score': float(selected_bottle['bottle_region_score']) if selected_bottle else 0.0,
        'box_region_score': float(selected_box['box_region_score']) if selected_box else 0.0,
        'depth_consistency_score': float(max(selected_bottle['depth_consistency_score'] if selected_bottle else 0.0, selected_box['depth_consistency_score'] if selected_box else 0.0)),
        'semantic_label_verified': sem_verified,
        'failure_reason': failure_reason,
        'auto_candidates_top_bottle': [{k: v for k, v in row.items() if k != 'mask'} for row in bottle_rows[:5]],
        'auto_candidates_top_box': [{k: v for k, v in row.items() if k != 'mask'} for row in box_rows[:5]],
    }
    return {
        'bottle': bottle_mask,
        'box': box_mask,
        'foreground': fg_mask,
        'background': bg_mask,
        'overlay_auto_masks': all_masks_overlay,
        'selected_bottle_mask': bottle_mask,
        'selected_box_mask': box_mask,
    }, diag


def main():
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    ds = LeRobotDataset(repo_id=REPO_ID, tolerance_s=0.001)
    sample0 = ds[0]
    resolved_image = IMAGE_KEY if IMAGE_KEY in sample0 else next((k for k in sample0.keys() if 'image' in k and 'depth' not in k), None)
    resolved_depth = DEPTH_KEY if DEPTH_KEY in sample0 else next((k for k in sample0.keys() if 'depth' in k), None)
    predictor, auto = sam_model_bundle()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        'repo_id': REPO_ID,
        'resolved_keys': {'image': resolved_image, 'depth': resolved_depth},
        'frames_tested': FRAMES,
        'methods': {},
        'frame_method_rows': [],
    }
    method_counts = {m: {'localization_pass_frames': 0, 'bottle_split_pass_frames': 0, 'box_split_pass_frames': 0, 'background_edit_pass_frames': 0, 'semantic_verified_frames': 0} for m in METHODS}

    for frame_index in FRAMES:
        sample = ds[frame_index]
        rgb = to_rgb_u8(to_np(sample[resolved_image]))
        depth = sanitize_depth(to_np(sample[resolved_depth]))[..., 0]
        valid = (depth > 0).astype(np.uint8) * 255
        result = build_rgbd_object_masks(rgb, depth, valid, frame_index=frame_index)
        heuristic_masks = predict_masks_for_method(result, rgb, 'grabcut')
        small_rgb = cv2.resize(rgb, (int(rgb.shape[1] * AUTO_SCALE), int(rgb.shape[0] * AUTO_SCALE)), interpolation=cv2.INTER_AREA)
        auto_masks_small = auto.generate(small_rgb)
        auto_masks = []
        for row in auto_masks_small:
            up = dict(row)
            up['segmentation'] = upscale_mask(np.asarray(row['segmentation']).astype(np.uint8) * 255, rgb.shape[:2]) > 0
            auto_masks.append(up)
        print(f'frame {frame_index}: auto_masks={len(auto_masks)}')
        fg_box = bbox_from_mask(heuristic_masks['foreground'])
        method_masks = {
            'seeded_grabcut': {**heuristic_masks, 'background': result.background_edit_mask, 'overlay_auto_masks': rgb.copy()},
            'sam_prompt_box': {
                'bottle': ensure_binary(best_sam_mask(predictor, rgb, bbox_from_mask(heuristic_masks['bottle']) or fg_box), valid),
                'box': ensure_binary(best_sam_mask(predictor, rgb, bbox_from_mask(heuristic_masks['box']) or fg_box), valid),
                'overlay_auto_masks': rgb.copy(),
            },
        }
        method_masks['sam_prompt_box']['box'] = cv2.subtract(method_masks['sam_prompt_box']['box'], cv2.dilate(method_masks['sam_prompt_box']['bottle'], np.ones((5, 5), np.uint8), iterations=1))
        method_masks['sam_prompt_box']['foreground'] = cv2.bitwise_and(cv2.bitwise_or(method_masks['sam_prompt_box']['bottle'], method_masks['sam_prompt_box']['box']), valid)
        method_masks['sam_prompt_box']['background'] = cv2.bitwise_and(cv2.bitwise_not(method_masks['sam_prompt_box']['foreground']), valid)

        auto_plain_masks, auto_plain_diag = score_auto_candidates(rgb, depth, valid, auto_masks, heuristic_masks, result.diagnostics.to_dict(), depth_gated=False)
        auto_depth_masks, auto_depth_diag = score_auto_candidates(rgb, depth, valid, auto_masks, heuristic_masks, result.diagnostics.to_dict(), depth_gated=True)
        method_masks['sam_auto_selector'] = auto_plain_masks
        method_masks['sam_auto_selector_depth_gated'] = auto_depth_masks
        extra_diags = {'sam_auto_selector': auto_plain_diag, 'sam_auto_selector_depth_gated': auto_depth_diag}

        for method, masks in method_masks.items():
            frame_dir = OUT_ROOT / method / f'frame_{frame_index:04d}'
            frame_dir.mkdir(parents=True, exist_ok=True)
            save_image(frame_dir / 'original.png', rgb)
            save_image(frame_dir / 'overlay_auto_masks.png', masks.get('overlay_auto_masks', rgb))
            save_image(frame_dir / 'overlay_selected_bottle.png', overlay_mask(rgb, masks['bottle'], (255, 0, 255)))
            save_image(frame_dir / 'overlay_selected_box.png', overlay_mask(rgb, masks['box'], (255, 255, 0)))
            overlay = overlay_mask(rgb, masks['foreground'], (0, 255, 255))
            overlay = overlay_mask(overlay, masks['bottle'], (255, 0, 255))
            overlay = overlay_mask(overlay, masks['box'], (255, 255, 0))
            save_image(frame_dir / 'overlay_instances_refined.png', overlay)
            save_image(frame_dir / 'bottle_mask.png', masks['bottle'])
            save_image(frame_dir / 'box_mask.png', masks['box'])
            save_image(frame_dir / 'background_edit_mask.png', masks['background'])

            loc_pass, loc_reason = localization_pass(masks['foreground'])
            bg_pass, bg_reason = background_edit_pass(masks['foreground'])
            sem_verified = bool(result.diagnostics.semantic_label_verified)
            bottle_pass = bool(sem_verified and mask_stats(masks['bottle'])['area'] > 1500)
            box_pass = bool(sem_verified and mask_stats(masks['box'])['area'] > 1500)
            reason_bits = []
            reason_bits.append('localized' if loc_pass else loc_reason)
            if not sem_verified:
                reason_bits.append('semantic_label_verified=false')
            else:
                if not bottle_pass:
                    reason_bits.append('bottle mask not reliable')
                if not box_pass:
                    reason_bits.append('box mask not reliable')
            if method in extra_diags and extra_diags[method].get('failure_reason'):
                reason_bits.append(extra_diags[method]['failure_reason'])
            diag = {
                'method': method,
                'frame_index': frame_index,
                'localization': 'pass' if loc_pass else 'fail',
                'bottle_split': 'pass' if bottle_pass else 'fail',
                'box_split': 'pass' if box_pass else 'fail',
                'background_edit': 'pass' if bg_pass else 'fail',
                'reason': '; '.join(reason_bits),
                'semantic_label_verified': sem_verified,
                'failure_reason': extra_diags.get(method, {}).get('failure_reason', result.diagnostics.failure_reason),
                'mask_stats': {k: mask_stats(v) for k, v in {'foreground': masks['foreground'], 'bottle': masks['bottle'], 'box': masks['box']}.items()},
                'background_area_ratio': float((masks['background'] > 0).sum() / masks['background'].size),
                'source_diagnostics': result.diagnostics.to_dict(),
            }
            if method in extra_diags:
                diag.update(extra_diags[method])
            (frame_dir / 'diagnostics.json').write_text(json.dumps(diag, ensure_ascii=False, indent=2))
            summary['frame_method_rows'].append({k: diag[k] for k in ['method', 'frame_index', 'localization', 'bottle_split', 'box_split', 'background_edit', 'reason', 'semantic_label_verified', 'failure_reason']})
            counts = method_counts[method]
            if loc_pass:
                counts['localization_pass_frames'] += 1
            if bottle_pass:
                counts['bottle_split_pass_frames'] += 1
            if box_pass:
                counts['box_split_pass_frames'] += 1
            if bg_pass:
                counts['background_edit_pass_frames'] += 1
            if sem_verified:
                counts['semantic_verified_frames'] += 1

    for method, counts in method_counts.items():
        summary['methods'][method] = {'method': method, **counts}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
