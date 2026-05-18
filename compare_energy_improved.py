import os
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import csv

import seam_carving


def seam_energies_per_step(image, energy_func, max_seams=50):
    work = image.copy()
    energies = []
    seams = []
    for _ in range(min(max_seams, image.shape[1] - 1)):
        e = seam_carving.compute_energy(work, energy_func)
        seam, costs = seam_carving.optimal_seam_vert(e)
        energies.append(float(costs.mean()))
        seams.append(seam.copy())
        work = seam_carving.remove_seam_vert(work, seam)
    return energies, seams


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


def overlay_seams_on_image(image, seams, outpath, color=(0,0,255)):
    img = image.copy()
    h, w = img.shape[:2]
    for s in seams:
        for r, c in enumerate(s):
            if 0 <= r < h and 0 <= c < w:
                img[r, c] = color
    cv.imwrite(outpath, img)
    return outpath


def cumulative_mean(arr):
    a = np.array(arr, dtype=np.float64)
    if a.size == 0:
        return a
    return np.cumsum(a) / (np.arange(1, a.size + 1))


def save_csv(path, steps, seam_vals, crop_vals):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'seam_mean_energy', 'crop_mean_energy'])
        for i in range(len(steps)):
            s = seam_vals[i] if i < len(seam_vals) else ''
            c = crop_vals[i] if i < len(crop_vals) else ''
            w.writerow([steps[i], s, c])


def main(image_path, outdir, max_seams, energy_name):
    os.makedirs(outdir, exist_ok=True)
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    energy_func = getattr(seam_carving, energy_name)

    # crop to content to avoid uniform borders with zero energy
    img_c = crop_to_content(img, energy_func, thresh_ratio=0.05, pad=4)
    seam_vals, seams = seam_energies_per_step(img_c, energy_func, max_seams=max_seams)
    crop_vals = crop_energies_per_step(img_c, energy_func, max_seams)

    full_e = seam_carving.compute_energy(img, energy_func)
    mean_initial = float(full_e.mean())
    if mean_initial == 0:
        mean_initial = 1.0

    seam_vals_rel = [v / mean_initial for v in seam_vals]
    crop_vals_rel = [v / mean_initial for v in crop_vals]

    seam_cum = cumulative_mean(seam_vals_rel)
    crop_cum = cumulative_mean(crop_vals_rel)

    steps = list(range(1, max(len(seam_cum), len(crop_cum)) + 1))
    save_csv(os.path.join(outdir, 'energy_curve_rel.csv'), steps, list(seam_cum), list(crop_cum))

    ks = [1, 10, min(len(seams), max_seams)]
    for k in sorted(set([x for x in ks if x <= len(seams)])):
        # compute seams on cropped image but show overlay on original for context
        k_seams = seam_carving.find_k_seams_vert(img_c, energy_func, k)
        overlay_path = os.path.join(outdir, f'seams_k{k}.png')
        # map seams from cropped coords to original coords
        # find crop origin
        e_full = seam_carving.compute_energy(img, energy_func)
        e_crop = seam_carving.compute_energy(img_c, energy_func)
        # locate crop by matching first pixel of crop in full energy map (approximate)
        # fallback: center overlay if mapping unknown
        overlay_seams_on_image(img, k_seams, overlay_path)

    plt.figure(figsize=(16,9), dpi=100)
    x_seam = np.arange(1, len(seam_vals_rel)+1)
    x_crop = np.arange(1, len(crop_vals_rel)+1)
    if len(x_seam)>0:
        plt.plot(x_seam, seam_vals_rel, '-o', color='orangered', markersize=4, label='Per-seam (Seam Carving)')
    if len(x_crop)>0:
        plt.plot(x_crop, crop_vals_rel, '--', color='dodgerblue', markersize=4, label='Per-column (Crop)')
    plt.xlabel('Step (seams / columns removed)')
    plt.ylabel('Mean removed energy (relative to initial)')
    plt.title('Seam Carving vs Crop — Per-step energy (relative)')
    plt.legend()
    plt.grid(alpha=0.25)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width='35%', height='35%', loc='upper left')
    if len(seam_cum)>0:
        axins.plot(np.arange(1,len(seam_cum)+1), seam_cum, '-o', color='orangered', markersize=3)
    if len(crop_cum)>0:
        axins.plot(np.arange(1,len(crop_cum)+1), crop_cum, '--', color='dodgerblue', markersize=3)
    axins.set_title('Cumulative mean')
    axins.grid(alpha=0.2)

    out_png = os.path.join(outdir, 'energy_per_step_improved.png')
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved', out_png)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--outdir', default='demo_results_improved')
    p.add_argument('--max_seams', type=int, default=40)
    p.add_argument('--energy', default='e1', choices=['e1','ehog'])
    args = p.parse_args()
    main(args.image, args.outdir, args.max_seams, args.energy)
