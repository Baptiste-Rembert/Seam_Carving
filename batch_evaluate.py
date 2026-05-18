import os
import csv
import argparse
import glob
import math
import matplotlib.pyplot as plt

from eval_metrics import evaluate


def find_images(indir):
    exts = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(indir, f'**/*.{e}'), recursive=True))
    # filter out files in .venv and results directories
    files = [f for f in files if '/.venv/' not in f and '/results' not in f and os.path.isfile(f)]
    return sorted(files)


def aggregate_metrics(metrics_list, csv_path):
    if not metrics_list:
        return
    # metrics_list is list of (image_path, metrics_dict)
    keys = list(metrics_list[0][1].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image'] + keys)
        w.writeheader()
        for name, m in metrics_list:
            row = {'image': os.path.basename(name)}
            # ensure all keys present
            for k in keys:
                row[k] = m.get(k, '')
            w.writerow(row)


def plot_metrics(csv_path, out_png):
    # read csv and plot numeric columns without pandas
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return
    images = [row['image'] for row in rows]
    # collect numeric columns
    numeric_cols = []
    for k in rows[0].keys():
        if k == 'image':
            continue
        try:
            # test convertibility on first row
            float(rows[0][k])
            numeric_cols.append(k)
        except Exception:
            continue
    if not numeric_cols:
        return
    n = len(numeric_cols)
    cols = 2
    rows_n = math.ceil(n / cols)
    fig, axs = plt.subplots(rows_n, cols, figsize=(12, 4 * rows_n))
    axs = axs.flatten()
    for i, colname in enumerate(numeric_cols):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[colname]))
            except Exception:
                vals.append(float('nan'))
        axs[i].bar(images, vals)
        axs[i].set_title(colname)
        axs[i].tick_params(axis='x', rotation=45)
    # hide any extra subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    fig.savefig(out_png)


def main(indir, outdir, scale):
    os.makedirs(outdir, exist_ok=True)
    imgs = find_images(indir)
    print(f'Found {len(imgs)} images')
    results = []
    for img in imgs:
        name = os.path.basename(img)
        print('Processing', name)
        # compute target size by scaling
        import cv2 as cv
        im = cv.imread(img)
        if im is None:
            print('  skip (not an image)')
            continue
        th = int(im.shape[0] * scale)
        tw = int(im.shape[1] * scale)
        out_sub = os.path.join(outdir, os.path.splitext(name)[0])
        try:
            m = evaluate(img, processed_path=None, target=(th, tw), energy='e1', outdir=out_sub)
            results.append((img, m))
        except Exception as e:
            print('  error evaluating', name, e)

    csv_path = os.path.join(outdir, 'batch_metrics.csv')
    aggregate_metrics(results, csv_path)
    try:
        plot_metrics(csv_path, os.path.join(outdir, 'metrics_plots.png'))
    except Exception as e:
        print('Could not plot metrics:', e)
    print('Batch done. CSV at', csv_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--indir', default='.', help='input images directory')
    p.add_argument('--outdir', default='batch_results', help='output directory for results')
    p.add_argument('--scale', type=float, default=0.8, help='scale factor for target size (0-1)')
    args = p.parse_args()
    main(args.indir, args.outdir, args.scale)
