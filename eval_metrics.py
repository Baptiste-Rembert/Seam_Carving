import os
import json
import argparse
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
from skimage.filters import rank
from skimage.morphology import disk
from skimage import img_as_ubyte

import seam_carving


def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)


def mean_gradient_magnitude(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sobelx, sobely)
    mean_val = float(np.mean(grad))
    # normalize for visualization
    vis = np.clip((grad / (grad.max() + 1e-9)) * 255, 0, 255).astype(np.uint8)
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    return mean_val, vis


def edge_density(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    density = float(edges.sum() / 255) / (edges.shape[0] * edges.shape[1])
    edges_vis = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    return density, edges_vis


def shannon_entropy_map(image, neighborhood=9):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_u8 = img_as_ubyte(gray)
    # use a disk-shaped neighborhood approximating 9x9 window
    ent = rank.entropy(gray_u8, disk(neighborhood // 2))
    mean_ent = float(np.mean(ent))
    # normalize for visualization
    vis = np.clip((ent / (ent.max() + 1e-9)) * 255, 0, 255).astype(np.uint8)
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
    return mean_ent, vis


def compute_ssim(a, b):
    # skimage ssim expects images in range [0,255]; use channel_axis for color
    score, diff = ssim(a, b, full=True, channel_axis=2)
    return float(score), (diff * 255).astype(np.uint8)


def keypoint_match_ratio(a, b, max_features=500):
    gray1 = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, 0
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m in matches if m.distance < 60]
    ratio = len(good) / max(1, min(len(kp1), len(kp2)))
    return float(ratio), len(good)


def seam_heatmap(original, reduced, energy_func, outpath):
    # compute how many seams removed in each direction
    h0, w0 = original.shape[:2]
    h1, w1 = reduced.shape[:2]
    dh = h0 - h1
    dw = w0 - w1
    heat = np.zeros((h0, w0), dtype=np.float32)
    # vertical seams
    if dw > 0:
        seams_v = seam_carving.find_k_seams_vert(original, energy_func, dw)
        for s in seams_v:
            heat[np.arange(h0), s] += 1
    # horizontal seams
    if dh > 0:
        seams_h = seam_carving.find_k_seams_hor(original, energy_func, dh)
        for s in seams_h:
            heat[s, np.arange(w0)] += 1
    # normalize
    if heat.max() > 0:
        nm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        nm = (heat).astype(np.uint8)
    cmap = cv.applyColorMap(nm, cv.COLORMAP_JET)
    overlay = cv.addWeighted(original, 0.6, cmap, 0.4, 0)
    cv.imwrite(outpath, overlay)
    return outpath


def evaluate(original_path, processed_path=None, target=None, energy='e1', outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    orig = cv.imread(original_path)
    if orig is None:
        raise FileNotFoundError(original_path)
    if processed_path is None and target is None:
        raise ValueError('Either provide processed_path or target (h,w)')

    if processed_path is None:
        th, tw = target
        proc = seam_carving.image_resize_down(orig, getattr(seam_carving, energy), th, tw)
    else:
        proc = cv.imread(processed_path)
        if proc is None:
            raise FileNotFoundError(processed_path)

    # resize processed to orig size if needed for some metrics (PSNR/SSIM/keypoints)
    proc_for_metrics = proc.copy()
    if proc_for_metrics.shape != orig.shape:
        proc_for_metrics = cv.resize(proc_for_metrics, (orig.shape[1], orig.shape[0]), interpolation=cv.INTER_LINEAR)

    # PSNR
    ps = compute_psnr(orig, proc_for_metrics)
    # SSIM
    ssim_score, ssim_map = compute_ssim(orig, proc_for_metrics)
    cv.imwrite(os.path.join(outdir, 'ssim_map.png'), ssim_map)
    # Keypoint matching
    kp_ratio, kp_count = keypoint_match_ratio(orig, proc_for_metrics)
    # Heatmap of seams
    heat_path = os.path.join(outdir, 'seams_overlay.png')
    try:
        seam_heatmap(orig, proc, getattr(seam_carving, energy), heat_path)
    except Exception:
        heat_path = None

    # Mean Gradient Magnitude
    mg_orig, mg_map_orig = mean_gradient_magnitude(orig)
    mg_proc, mg_map_proc = mean_gradient_magnitude(proc_for_metrics)
    cv.imwrite(os.path.join(outdir, 'mean_gradient_orig.png'), mg_map_orig)
    cv.imwrite(os.path.join(outdir, 'mean_gradient_proc.png'), mg_map_proc)

    # Edge Density
    ed_orig, ed_map_orig = edge_density(orig)
    ed_proc, ed_map_proc = edge_density(proc_for_metrics)
    cv.imwrite(os.path.join(outdir, 'edges_orig.png'), ed_map_orig)
    cv.imwrite(os.path.join(outdir, 'edges_proc.png'), ed_map_proc)

    # Shannon Entropy (local)
    ent_orig, ent_map_orig = shannon_entropy_map(orig, neighborhood=9)
    ent_proc, ent_map_proc = shannon_entropy_map(proc_for_metrics, neighborhood=9)
    cv.imwrite(os.path.join(outdir, 'entropy_orig.png'), ent_map_orig)
    cv.imwrite(os.path.join(outdir, 'entropy_proc.png'), ent_map_proc)

    metrics = {
        'psnr': ps,
        'ssim': ssim_score,
        'keypoint_match_ratio': kp_ratio,
        'keypoint_matches': kp_count,
        'seams_overlay': heat_path,
        'mean_gradient_orig': mg_orig,
        'mean_gradient_proc': mg_proc,
        'edge_density_orig': ed_orig,
        'edge_density_proc': ed_proc,
        'entropy_mean_orig': ent_orig,
        'entropy_mean_proc': ent_proc,
    }
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Metrics:', json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('original')
    p.add_argument('--processed', '-p', help='Processed image path (optional)')
    p.add_argument('--target', '-t', help='target size HxW, e.g. 300x400')
    p.add_argument('--energy', default='e1', choices=['e1', 'ehog'])
    p.add_argument('--outdir', default='results')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    target = None
    if args.target:
        try:
            th, tw = args.target.split('x')
            target = (int(th), int(tw))
        except:
            raise ValueError('target must be HxW')
    evaluate(args.original, processed_path=args.processed, target=target, energy=args.energy, outdir=args.outdir)
