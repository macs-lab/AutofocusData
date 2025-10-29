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
# Color scheme used:
COLOR = ["#E69F00", "#009E73","#56B4E9"]
LINESTYLE = ['--', ':', '-']

def find_steel_ehc_dirs(root):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith(PREFIX) and os.path.isdir(os.path.join(root, d))
    ]

def find_alg_dirs(root, contain):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if contain in d.lower() and os.path.isdir(os.path.join(root, d))
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

def read_csv_focus_data(filename, max_fv=0.5e6):
    dema, dfv, ddfv, ratio = [], [], [], []
    times_raw, x_raw = [], []
    stop_flag = False

    def safe_val(v):
        return 0 if v is None or (isinstance(v, float) and math.isnan(v)) else v

    with open(filename, newline='', encoding='utf-8') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            if 'return to max' in row[20].lower():
                stop_flag = True
                break

            fv = safe_float(row[8].strip())
            if fv is None:
                continue
            fv = abs(fv)

            dfv_v = safe_float(row[9].strip())
            ddfv_v = safe_float(row[10].strip())
            ratio_v = safe_float(row[11].strip())
            t_raw = safe_float(row[12].strip())

            # Read x, y, z for Euclidean norm
            x_val = safe_float(row[12].strip())
            y_val = safe_float(row[13].strip())
            z_val = safe_float(row[14].strip())
            x_safe = safe_val(x_val)
            y_safe = safe_val(y_val)
            z_safe = safe_val(z_val)
            x_raw_v = math.sqrt(x_safe**2 + y_safe**2 + z_safe**2)

            dema.append(fv)
            dfv.append(dfv_v if dfv_v is not None else float('nan'))
            ddfv.append(ddfv_v if ddfv_v is not None else float('nan'))
            ratio.append(ratio_v if ratio_v is not None else float('nan'))
            times_raw.append(t_raw if t_raw is not None else float('nan'))
            x_raw.append(x_raw_v if x_raw_v is not None else float('nan'))

    # convert timestamps: if any value looks like nanoseconds (>1e6) convert to seconds
    convert_ns = any((v is not None and not math.isnan(v) and abs(v) > 1e6) for v in times_raw)
    times = [ (v * 1e-9) if (not math.isnan(v) and convert_ns) else (v if not math.isnan(v) else float('nan')) for v in times_raw ]

    # normalize time to start at 0 using first valid timestamp
    baseline = next((t for t in times if not math.isnan(t)), 0.0)
    time_norm = [ (t - baseline) if not math.isnan(t) else float('nan') for t in times ]

    # x: shift raw x so the first valid x becomes 0, and ensure x moves in positive direction
    first_x = next((v for v in x_raw if not (v is None or math.isnan(v))), None)
    if first_x is None:
        x_vals = [float('nan')] * len(x_raw)
    else:
        shifted = [ (v - first_x) if not (v is None or math.isnan(v)) else float('nan') for v in x_raw ]
        last_valid = next((v for v in reversed(shifted) if not (v is None or math.isnan(v))), None)
        if last_valid is not None and last_valid < 0:
            # flip sign so direction is positive
            x_vals = [(-v if not (v is None or math.isnan(v)) else float('nan')) for v in shifted]
        else:
            x_vals = shifted

    # shift x so that x = 0.043 becomes the new zero (x' = x - 0.043)
    offset = 0.043
    x_vals = [ (xx - offset) if not (xx is None or math.isnan(xx)) else float('nan') for xx in x_vals ]

    # replace dema outliers (> max_fv) with fixed value
    replacement = 250000
    for i, v in enumerate(dema):
        if v > max_fv:
            dema[i] = replacement

    return time_norm, dema, dfv, ddfv, ratio, x_vals, stop_flag

def clean_metric_name(dirname):
    basename = os.path.basename(dirname)
    metric = re.sub(r'^Steel_ehc_', '', basename)
    metric = re.sub(r'[_0-9]+$', '', metric)
    return metric

def moving_average(values, window=5):
    # Simple moving average that returns a list the same length as `values`.
    # Change window size to adjust smoothing (odd numbers center better)
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

def plot_3_metrics(steel_data):
    if not steel_data:
        print("No data to plot.")
        return
    
    # Plot Focus Value vs X for all metrics
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)    
    for i, (metric_name, metric_data) in enumerate(steel_data.items()):
        ax.plot(
            metric_data["x"],
            metric_data["dema_fv"],
            label=metric_name,
            color=COLOR[i],
            linestyle=LINESTYLE[i],
            linewidth=1
        )

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Focus Value", fontsize=9)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Focus Value Across Metrics vs Position X", fontsize=9)
    plt.tight_layout()
    plt.xlim(0,0.05)
    plt.savefig("fv_comparison_vibrant.png")
    plt.show()

    # Plot Ratio vs X for all metrics
    # fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    # for metric_name, metric_data in steel_data.items():
    #     ax.plot(metric_data["x"], metric_data["ratio"], label=metric_name, linewidth=1)
    # ax.set_xlabel("X (m)", fontsize=10)
    # ax.set_ylabel("Ratio", fontsize=10)
    # ax.legend(fontsize=9)
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    # ax.tick_params(axis='x', labelsize=8)
    # ax.tick_params(axis='y', labelsize=8)
    # ax.set_title("Ratio Across Metrics vs Position X", fontsize=12)
    # plt.tight_layout()
    # plt.xlim(0,0.05)
    # plt.show()

    # Plot Smoothed Ratio vs X for all metrics. Simple Moving Average works fine. Don't use EMA since SMA is better for noise and smoothing.
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)    
    for i, (metric_name, metric_data) in enumerate(steel_data.items()):
        x_vals = metric_data["x"]
        ratio_vals = metric_data["ratio"]
        # compute and plot moving average
        smoothed = moving_average(ratio_vals, window=11)
        ax.plot(x_vals, smoothed, linewidth=1, label=f"{metric_name}", color=COLOR[i], linestyle=LINESTYLE[i])

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Ratio", fontsize=9)
    ax.legend(fontsize=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Ratio Across Metrics vs Position X", fontsize=9)
    plt.tight_layout()
    plt.xlim(0,0.05)
    plt.savefig("ratio_comparison.png")
    plt.show()

def plot_dfv_ddfv(data):
    if not data:
        print("No data to plot.")
        return

    x = data["x"]
    dfv = data["dfv"]
    ddfv = data["ddfv"]
    ratio = data["ratio"]
    dema = data["dema_fv"]

    fig, ax = plt.subplots(4, 1, figsize=(3.5, 6.5), dpi=300, sharex=True)
    ax[0].plot(x, dema, linewidth=1, color="#56B4E9")
    ax[0].set_xlim(0,0.05)
    ax[0].set_ylabel("Focus Value", fontsize=9)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=6)); ax[0].yaxis.set_major_locator(MaxNLocator(nbins=4))

    smoothed_dfv = moving_average(dfv, window=5)
    ax[1].plot(x, smoothed_dfv, linewidth=1, color="#56B4E9")
    ax[1].set_xlim(0,0.05)
    ax[1].set_ylabel("DFV", fontsize=9)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=6)); ax[1].yaxis.set_major_locator(MaxNLocator(nbins=4))

    smoothed_ddfv = moving_average(ddfv, window=5)
    ax[2].plot(x, smoothed_ddfv, linewidth=1, color="#56B4E9")
    ax[2].set_xlim(0,0.05)
    ax[2].set_ylabel("DDFV", fontsize=9)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=6)); ax[2].yaxis.set_major_locator(MaxNLocator(nbins=4))

    smoothed_ratio = moving_average(ratio, window=11)
    ax[3].plot(x, smoothed_ratio, linewidth=1, color="#56B4E9")
    ax[3].set_xlim(0,0.05)
    ax[3].set_xlabel("X (m)", fontsize=9); ax[3].set_ylabel("Ratio", fontsize=9)
    ax[3].xaxis.set_major_locator(MaxNLocator(nbins=6)); ax[3].yaxis.set_major_locator(MaxNLocator(nbins=4))

    fig.suptitle("            FV, DFV, DDFV, Ratio vs Position X", fontsize=9)

    # find first index where dfv > 0 and ddfv < 0
    mark_idx = None
    for i, (d1, d2, xv) in enumerate(zip(smoothed_dfv, smoothed_ddfv, x)):
        try:
            if xv is None or math.isnan(xv): 
                continue
            if d1 is None or math.isnan(d1) or d2 is None or math.isnan(d2):
                continue
            if d1 > 0.1 and d2 < -0.1:
                mark_idx = i
                break
        except Exception:
            continue

    if mark_idx is not None:
        x_mark = x[mark_idx]
        # vertical line across all subplots
        for a in ax:
            a.axvline(x_mark, color="#E69F00", linestyle='--', linewidth=1, label="Fine mode starts")
    ax[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("FV_dFV_ddFV.png")
    plt.show()

def read_final_time(filename):
    # Final time is the last timestamp before "return to max" focus mode
    baseline = None
    final_ns = None

    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row

        for row in reader:
            focus_mode = row[-2].lower()
            if "return to max" in focus_mode:
                break

            raw_time_str = row[0].strip()
            raw_time_ns = float(raw_time_str)
            if baseline is None:
                baseline = raw_time_ns
            final_ns = raw_time_ns
    return (final_ns - baseline) * 1e-9  # normalize and convert to seconds

def write_to_csv(output_filename, filename, value1, value2=None):
    with open(output_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([filename, value1, value2])# if value2 is None else [filename, value1, value2])


def main():
    # Plotting graphs for Methods section
    steel_dir = find_steel_ehc_dirs(ROOT)
    if not steel_dir:
        print("No Steel_ehc directories found under", ROOT)
        return
    
    steel_data = {}
    for d in steel_dir:
        csv_path = first_csv_in_dir(d)
        if csv_path:
            time, dema_fv, dfv, ddfv, ratio, x, stop_flag = read_csv_focus_data(csv_path)
            metric_name = clean_metric_name(d)
            # only add if we actually read some data
            if dema_fv:
                steel_data[metric_name] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
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
    plot_3_metrics(steel_data)
    plot_dfv_ddfv(steel_data["sobel"])

    # Time and position processing for all EHC and Adaptive runs
    ehc_dirs = find_alg_dirs(ROOT,'ehc')
    adaptive_dirs = find_alg_dirs(ROOT, 'adaptive') + find_alg_dirs(ROOT, 'default')

    # EHC runs
    all_data = {}
    for d in ehc_dirs:
        csv_path = first_csv_in_dir(d)
        if csv_path:
            # Final time computation + write to CSV
            time_taken = read_final_time(csv_path)
            write_to_csv('Time_Taken.csv', os.path.basename(d), time_taken)

            # Max X focus computatoin
            time, dema_fv, dfv, ddfv, ratio, x, stop_flag = read_csv_focus_data(csv_path)
            if dema_fv:
                all_data[os.path.basename(d)] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
                    "ratio": ratio,
                    "x": x
                }
                max_focus = max(dema_fv)
                max_index = dema_fv.index(max_focus)
                max_focus_x = x[max_index]
                write_to_csv('Max_Focus_X.csv', os.path.basename(d), max_focus_x, max_focus)
            else:
                print(f"No valid data read from ({os.path.basename(csv_path)}).")
            if stop_flag:
                continue
        else:
            print(f"No .csv file found in {d}")

    # Adaptive runs
    all_data = {}
    for d in adaptive_dirs:
        csv_path = first_csv_in_dir(d)
        if csv_path:
            # Final time computation + write to CSV
            time_taken = read_final_time(csv_path)
            write_to_csv('Time_Taken.csv', os.path.basename(d), time_taken)

            # Max X focus computatoin
            time, dema_fv, dfv, ddfv, ratio, x, stop_flag = read_csv_focus_data(csv_path)
            if dema_fv:
                all_data[os.path.basename(d)] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
                    "ratio": ratio,
                    "x": x
                }
                max_focus = max(dema_fv)
                max_index = dema_fv.index(max_focus)
                max_focus_x = x[max_index]
                write_to_csv('Max_Focus_X.csv', os.path.basename(d), max_focus_x, max_focus)
            else:
                print(f"No valid data read from ({os.path.basename(csv_path)}).")
            if stop_flag:
                continue
        else:
            print(f"No .csv file found in {d}")

if __name__ == "__main__":
    main()