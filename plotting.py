import os
import csv
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import math

# Plotting 3 focus metrics together, first as focus value, then as ratio
ROOT = os.path.dirname(__file__)
PREFIX = "Steel_ehc"

def find_steel_ehc_dirs(root):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith(PREFIX) and os.path.isdir(os.path.join(root, d))
    ]

def first_csv_in_dir(d):
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".csv"):
            return os.path.join(d, name)
    return None

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None

def read_csv_data(filename, max_fv=0.5e6):
    dema_fv, ratio, x = [], [], []
    stop_flag = False
    with open(filename, newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            # check column 21 (index 20) for stopping condition
            if 'return to max' in row[20].lower():
                stop_flag = True
                print(f"Found 'return to max' in column 21 of {os.path.basename(filename)} — stopping read of this file.")
                break

            fv_raw = row[8].strip()
            # also keep original check on focus column
            if fv_raw.lower() == 'adaptive return to max':
                continue

            fv = safe_float(fv_raw)
            if fv is None:
                continue
            fv = abs(fv)

            # safe access for ratio and x
            rx = None
            xx = None
            try:
                rx = safe_float(row[11].strip())
            except Exception:
                rx = None
            try:
                xx = safe_float(row[12].strip())
            except Exception:
                xx = None

            # fallback for x: use current index if column x not numeric
            if xx is None:
                xx = float(len(dema_fv))
            else:
                # shift x by 0.26 as requested
                xx = xx - 0.25

            dema_fv.append(fv)
            ratio.append(rx if rx is not None else float('nan'))
            x.append(xx)

    # replace outliers in dema_fv with a fixed value (250000) and keep alignment
    if dema_fv:
        replacement_value = 250000
        replaced = 0
        new_dema, new_ratio, new_x = [], [], []
        for fv, r, xx in zip(dema_fv, ratio, x):
            if fv > max_fv:
                fv = replacement_value
                replaced += 1
            new_dema.append(fv)
            new_ratio.append(r)
            new_x.append(xx)
        if replaced:
            print(f"Replaced {replaced} outlier(s) in '{os.path.basename(filename)}' with {replacement_value} (threshold {max_fv}).")
        return new_dema, new_ratio, new_x, stop_flag

    return dema_fv, ratio, x, stop_flag

def clean_metric_name(dirname):
    basename = os.path.basename(dirname)
    metric = re.sub(r'^Steel_ehc_', '', basename)
    metric = re.sub(r'[_0-9]+$', '', metric)
    return metric

def moving_average(values, window=5):
    # Simple moving average that returns a list the same length as `values`.
    if window <= 1:
        return list(values)
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        s = 0.0
        count = 0
        for v in values[start:end]:
            if v is None:
                continue
            # treat NaN-like entries
            try:
                if math.isnan(v):
                    continue
            except Exception:
                pass
            s += float(v)
            count += 1
        out.append(s / count if count > 0 else math.nan)
    return out

def plot_all_metrics(all_data):
    if not all_data:
        print("No data to plot.")
        return
    
    # Plot Focus Value vs X for all metrics
    fig, ax = plt.subplots(figsize=(10,6))
    for metric_name, metric_data in all_data.items():
        ax.plot(metric_data["x"], metric_data["dema_fv"], label=metric_name, linewidth=1.5)
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Focus Value", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Focus Value Across Metrics vs Position X", fontsize=12)
    plt.tight_layout()
    plt.xlim(0,0.05)
    plt.show()

    # Plot Ratio vs X for all metrics
    fig, ax = plt.subplots(figsize=(10,6))
    for metric_name, metric_data in all_data.items():
        ax.plot(metric_data["x"], metric_data["ratio"], label=metric_name, linewidth=1.5)
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Ratio", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Ratio Across Metrics vs Position X", fontsize=12)
    plt.tight_layout()
    plt.xlim(0,0.05)
    plt.show()

    # Plot Smoothed Ratio vs X for all metrics. Simple Moving Average works fine. Don't use EMA since SMA is better for noise and smoothing.
    fig, ax = plt.subplots(figsize=(10,6))
    window_size = 11  # change this to adjust smoothing (odd numbers center better)
    for metric_name, metric_data in all_data.items():
        x_vals = metric_data["x"]
        ratio_vals = metric_data["ratio"]
        # compute and plot moving average
        smoothed = moving_average(ratio_vals, window=window_size)
        ax.plot(x_vals, smoothed, linewidth=1.5, label=f"{metric_name}")

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Ratio", fontsize=10)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Ratio Across Metrics vs Position X", fontsize=12)
    plt.tight_layout()
    plt.xlim(0,0.05)
    plt.show()

def main():
    dirs = find_steel_ehc_dirs(ROOT)
    if not dirs:
        print("No Steel_ehc directories found under", ROOT)
        return
    
    all_data = {}
    for d in dirs:
        csv_path = first_csv_in_dir(d)
        if csv_path:
            dema_fv, ratio, x, stop_flag = read_csv_data(csv_path)
            metric_name = clean_metric_name(d)
            # only add if we actually read some data
            if dema_fv:
                all_data[metric_name] = {
                    "dema_fv": dema_fv,
                    "ratio": ratio,
                    "x": x
                }
            else:
                print(f"No valid data read from {metric_name} ({os.path.basename(csv_path)}).")
            if stop_flag:
                # print("Found 'return to max' in this file — stopping reading further rows of this file and continuing to next folder.")
                continue
        else:
            print(f"No .csv file found in {d}")

    # plot once for all metrics
    plot_all_metrics(all_data)

if __name__ == "__main__":
    main()