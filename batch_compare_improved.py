import os
import argparse
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import csv

import seam_carving


def find_images(indir):
    exts = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(indir, f'**/*.{e}'), recursive=True))
    files = [f for f in files if '/.venv/' not in f and '/demo_results' not in f and '/demo_results_improved' not in f and '/batch_results' not in f and os.path.isfile(f)]
    return sorted(files)


def seam_energies_per_step(image, energy_func, max_seams=50):
    work = image.copy()
    energies = []
    for _ in range(min(max_seams, image.shape[1] - 1)):
        e = seam_carving.compute_energy(work, energy_func)
        seam, costs = seam_carving.optimal_seam_vert(e)
        energies.append(float(costs.mean()))
        work = seam_carving.remove_seam_vert(work, seam)
    return energies


def crop_to_content(image, energy_func, thresh_ratio=0.05, pad=5):
    e = seam_carving.compute_energy(image, energy_func)
    mx = e.max()
    if mx <= 0:
        return image
    thresh = mx * thresh_ratio
    mask = e > thresh
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0]-1, y1 + pad)
    x1 = min(image.shape[1]-1, x1 + pad)
    return image[y0:y1+1, x0:x1+1]


def crop_energies_per_step(image, energy_func, max_steps=50):
    h, w = image.shape[:2]
    full_e = seam_carving.compute_energy(image, energy_func)
    energies = []
    for k in range(1, min(max_steps, w - 1) + 1):
        left = k // 2
        right = k - left
        cols_left = list(range(0, left))
        cols_right = list(range(w - right, w)) if right > 0 else []
        cols = cols_left + cols_right
        if len(cols) == 0:
            energies.append(0.0)
            continue
        removed_vals = full_e[:, cols]
        energies.append(float(removed_vals.mean()))
    return energies


def cumulative_mean(arr):
    a = np.array(arr, dtype=np.float64)
    if a.size == 0:
        return a
    return np.cumsum(a) / (np.arange(1, a.size + 1))


def run_batch(indir, outdir, max_steps, energy_name):
    os.makedirs(outdir, exist_ok=True)
    imgs = find_images(indir)
    print('Found', len(imgs), 'images')
    if not imgs:
        print('No images found')
        return
    energy_func = getattr(seam_carving, energy_name)

    K = max_steps
    seam_matrix = []
    crop_matrix = []
    valid_names = []
    for img_path in imgs:
        try:
            img = cv.imread(img_path)
            if img is None:
                continue
            # crop to content to avoid zero-energy borders
            img_c = crop_to_content(img, energy_func, thresh_ratio=0.05, pad=4)
            seam = seam_energies_per_step(img_c, energy_func, max_seams=K)
            crop = crop_energies_per_step(img_c, energy_func, max_steps=K)
            # normalize by mean initial energy
            full_e = seam_carving.compute_energy(img, energy_func)
            mean_init = float(full_e.mean())
            if mean_init == 0:
                mean_init = 1.0
            seam_rel = [v / mean_init for v in seam]
            crop_rel = [v / mean_init for v in crop]
            # pad to length K with np.nan
            seam_arr = np.full(K, np.nan)
            crop_arr = np.full(K, np.nan)
            seam_arr[:len(seam_rel)] = seam_rel
            crop_arr[:len(crop_rel)] = crop_rel
            seam_matrix.append(seam_arr)
            crop_matrix.append(crop_arr)
            valid_names.append(os.path.basename(img_path))
            print('  processed', os.path.basename(img_path))
        except Exception as e:
            print('  skip', img_path, e)

    seam_matrix = np.array(seam_matrix)
    crop_matrix = np.array(crop_matrix)

    # compute mean and 95% CI across images at each step (nan-aware)
    def mean_ci(mat):
        m = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        n = np.sum(~np.isnan(mat), axis=0)
        se = sd / np.sqrt(np.maximum(n, 1))
        ci = 1.96 * se
        return m, m - ci, m + ci, n

    seam_m, seam_lo, seam_hi, seam_n = mean_ci(seam_matrix)
    crop_m, crop_lo, crop_hi, crop_n = mean_ci(crop_matrix)

    steps = np.arange(1, K+1)
    # save CSV
    csv_path = os.path.join(outdir, 'energy_mean_ci.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step','seam_mean','seam_lo','seam_hi','seam_n','crop_mean','crop_lo','crop_hi','crop_n'])
        for i in range(K):
            w.writerow([i+1, seam_m[i] if not np.isnan(seam_m[i]) else '', seam_lo[i] if not np.isnan(seam_lo[i]) else '', seam_hi[i] if not np.isnan(seam_hi[i]) else '', int(seam_n[i]), crop_m[i] if not np.isnan(crop_m[i]) else '', crop_lo[i] if not np.isnan(crop_lo[i]) else '', crop_hi[i] if not np.isnan(crop_hi[i]) else '', int(crop_n[i])])

    # plot mean with shaded CI
    plt.figure(figsize=(16,9), dpi=120)
    valid = ~np.isnan(seam_m)
    if valid.any():
        plt.plot(steps[valid], seam_m[valid], color='orangered', label='Seam Carving (mean)')
        plt.fill_between(steps[valid], seam_lo[valid], seam_hi[valid], color='orangered', alpha=0.2)
    valid2 = ~np.isnan(crop_m)
    if valid2.any():
        plt.plot(steps[valid2], crop_m[valid2], color='dodgerblue', label='Crop (mean)')
        plt.fill_between(steps[valid2], crop_lo[valid2], crop_hi[valid2], color='dodgerblue', alpha=0.2)
    plt.xlabel('Step (seams / columns removed)')
    plt.ylabel('Mean removed energy (relative to initial)')
    plt.title('Mean per-step energy ± 95% CI across images')
    plt.legend()
    plt.grid(alpha=0.2)
    out_png = os.path.join(outdir, 'energy_mean_ci.png')
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved', out_png, 'and', csv_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--indir', default='.', help='input directory')
    p.add_argument('--outdir', default='demo_mean_ci')
    p.add_argument('--max_steps', type=int, default=30)
    p.add_argument('--energy', default='e1', choices=['e1','ehog'])
    args = p.parse_args()
    run_batch(args.indir, args.outdir, args.max_steps, args.energy)
