import os
import csv
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import math

# Plotting 3 focus metrics together, first as focus value, then as ratio
ROOT = os.path.dirname(__file__)
PREFIX = "Steel_ehc"
# Color scheme used:
COLOR = ["#E69F00", "#009E73","#56B4E9","#CC79A7"]
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

def read_csv_focus_data(filename, offset=0, max_fv=0.5e6):
    dema, dfv, ddfv, ratio, velocity = [], [], [], [], []
    times_raw, x_raw = [], []
    modes = []
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
            velocity_v = safe_float(row[19].strip())
            # Timestamp is in the first column (TIME_COL = 0). Previously this used
            # column 12 which is actually the X coordinate; that caused time to
            # be mixed with position and produced negative normalized times.
            t_raw = safe_float(row[0].strip())

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
            velocity.append(velocity_v if velocity_v is not None else float('nan'))
            times_raw.append(t_raw if t_raw is not None else float('nan'))
            x_raw.append(x_raw_v if x_raw_v is not None else float('nan'))
            # read the 2nd-to-last column (mode/state) if present
            try:
                mode_val = row[-2].strip()
            except Exception:
                mode_val = ""
            modes.append(mode_val)

    # convert timestamps: detect if values are in nanoseconds and convert to seconds.
    # Use a large threshold (1e12) to avoid mis-detecting small position values.
    convert_ns = any((v is not None and not math.isnan(v) and abs(v) > 1e12) for v in times_raw)
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

    # shift x so that x = 0.number becomes the new zero (x' = x - number.043)
    x_vals = [ (xx - offset) if not (xx is None or math.isnan(xx)) else float('nan') for xx in x_vals ]

    # replace dema outliers (> max_fv) with fixed value
    replacement = 250000
    for i, v in enumerate(dema):
        if v > max_fv:
            dema[i] = replacement

    return time_norm, dema, dfv, ddfv, ratio, velocity, x_vals, modes, stop_flag

def clean_metric_name(dirname):
    # get basename, normalize separators
    b = os.path.basename(dirname)
    name = re.sub(r'[_\-]+', ' ', b).strip()
    tokens = [t for t in name.split() if t]
    if not tokens:
        return b
    # tokens to ignore (common prefixes/labels)
    ignore = {'steel', 'steel_ehc', 'ehc', 'default', 'adaptive', 'run', 'method', 'pcb', 'cf'}
    # pick last token that is not numeric/version and not in ignore
    for t in reversed(tokens):
        tl = t.lower()
        if re.match(r'^v?\d+$', tl):
            continue
        if tl in ignore:
            continue
        return tl  # return lowercase short metric name
    # fallback: last token
    return tokens[-1].lower()

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
        mode_marker_added = False
        for a in ax:
            if not mode_marker_added:
                a.axvline(x_mark, color=COLOR[3], linestyle='--', linewidth=0.5, label="Fine mode starts")
                mode_marker_added = True
            else:
                a.axvline(x_mark, color=COLOR[3], linestyle='--', linewidth=0.5)
    ax[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("FV_dFV_ddFV.png")
    plt.show()

def plot_1_obj(data, dataset_name):
    if not data:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    mode_candidates = []
    for i, (metric_name, metric_data) in enumerate(data.items()):
        label = clean_metric_name(metric_name)
        x = metric_data.get("x", [])
        fv = metric_data.get("dema_fv", [])

        # compute first switch from 'coarse' -> 'fine' using the 2nd-to-last column values
        modes = metric_data.get("mode", [])
        mark_idx = None
        try:
            for j in range(1, min(len(modes), len(x))):
                prev = (modes[j-1] or '').lower()
                cur = (modes[j] or '').lower()
                if 'coarse' in prev and 'fine' in cur:
                    mark_idx = j
                    break
        except Exception:
            mark_idx = None

        ax.plot(
            x,
            fv,
            label=label,
            color=COLOR[i],
            linestyle=LINESTYLE[i],
            linewidth=1
        )

        # collect candidate mark positions (do not draw here)
        if mark_idx is not None and 0 <= mark_idx < len(x):
            try:
                x_mark = x[mark_idx]
                if x_mark is not None and not (isinstance(x_mark, float) and math.isnan(x_mark)):
                    mode_candidates.append(x_mark)
            except Exception:
                pass

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Focus Value", fontsize=9)
    # After plotting metrics, draw a single vertical marker at the smallest-magnitude candidate (if any)
    if mode_candidates:
        # choose the candidate closest to zero (smallest absolute value)
        chosen = min(mode_candidates, key=lambda v: abs(v))
        ax.axvline(chosen, color=COLOR[3], linestyle='-.', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    # append the mode marker to the end of the legend (if present)
    if mode_candidates:
        uniq_h.append(Line2D([0], [0], color=COLOR[3], linestyle='-.', linewidth=0.5))
        uniq_l.append("Fine mode starts")
    if uniq_h:
        ax.legend(uniq_h, uniq_l, fontsize=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f"FV vs X plot for {dataset_name}" if dataset_name else "FV vs X", fontsize=9)
    plt.tight_layout()
    # plt.xlim(0,0.05)
    plt.savefig("fv_unsmoothed.png")
    plt.show()

def compute_shifted_x(metric_data, target_x=None):
    x = metric_data.get("x", [])
    fv = metric_data.get("dema_fv", [])
    if target_x is None or not fv or not x:
        return x
    try:
        arr = np.array(fv, dtype=float)
        idx = int(np.nanargmax(arr))
    except Exception:
        return x
    if idx < 0 or idx >= len(x):
        return x
    peak_x = x[idx]
    try:
        shift = target_x - peak_x
    except Exception:
        return x
    return [(xx + shift) for xx in x]

def find_mode_switch_x(metric_data, x_vals):
    # Return the x coordinate for the first coarse->fine switch
    modes = metric_data.get("mode", [])
    for j in range(1, min(len(modes), len(x_vals))):
        prev = (modes[j-1] or '').lower()
        cur = (modes[j] or '').lower()
        if 'coarse' in prev and 'fine' in cur:
            return x_vals[j]
    return None

def plot_1_metric(all_data, metric_token, title=None):
    if not all_data:
        print("No data available to plot.")
        return

    token = metric_token.lower()
    # enforce legend order: CF, Steel, PCB
    materials = [("CF", "cf"), ("Steel", "steel"), ("PCB", "pcb")]
    selected = {}
    for label, mat_tok in materials:
        for k, v in all_data.items():
            kn = k.lower()
            if mat_tok in kn and token in kn:
                selected[label] = v
                break

    if not selected:
        print(f"No runs found for metric '{metric_token}'.")
        return
    
    # FV vs X
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    fv_mode_candidates = []
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        fv = metric_data.get("dema_fv", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        ax.plot(
            shifted_x,
            fv,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )

        # detect coarse->fine switch using helper
        m = find_mode_switch_x(metric_data, shifted_x)
        if m is not None:
            fv_mode_candidates.append(m)

    # draw only the smallest-magnitude FV mode marker (if any) and append legend last
    if fv_mode_candidates:
        chosen = min(fv_mode_candidates, key=lambda v: abs(v))
        ax.axvline(chosen, color=COLOR[3], linestyle='-.', linewidth=0.5)

    # dedupe legend just in case, and append the mode marker as the last legend entry
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if fv_mode_candidates:
        uniq_h.append(Line2D([0], [0], color=COLOR[3], linestyle='-.', linewidth=0.5))
        uniq_l.append("Fine mode starts")
    if uniq_h:
        ax.legend(uniq_h, uniq_l, fontsize=6)

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("FV", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f"FV vs X plot for {title or metric_token}", fontsize=9)
    plt.tight_layout()
    plt.xlim(0, 0.05)
    plt.savefig(f"fv_{metric_token}.png")
    plt.show()

    # Ratio vs X
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    ratio_mode_candidates = []
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        ratio = metric_data.get("ratio", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        ax.plot(
            shifted_x,
            ratio,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )

        m = find_mode_switch_x(metric_data, shifted_x)
        if m is not None:
            ratio_mode_candidates.append(m)

    # draw only the smallest-magnitude Ratio mode marker and append legend last
    if ratio_mode_candidates:
        chosen = min(ratio_mode_candidates, key=lambda v: abs(v))
        ax.axvline(chosen, color=COLOR[3], linestyle='-.', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if ratio_mode_candidates:
        uniq_h.append(Line2D([0], [0], color=COLOR[3], linestyle='-.', linewidth=0.5))
        uniq_l.append("Fine mode starts")
    if uniq_h:
        ax.legend(uniq_h, uniq_l, fontsize=6)

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Ratio", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f"Ratio vs X plot for {title or metric_token}", fontsize=9)
    plt.tight_layout()
    plt.xlim(0, 0.05)
    plt.savefig(f"ratio_{metric_token}.png")
    plt.show()

    # Velocity vs X
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    vel_mode_candidates = []
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        vel = metric_data.get("velocity", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        ax.plot(
            shifted_x,
            vel,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )

        m = find_mode_switch_x(metric_data, shifted_x)
        if m is not None:
            vel_mode_candidates.append(m)

    if vel_mode_candidates:
        chosen = min(vel_mode_candidates, key=lambda v: abs(v))
        ax.axvline(chosen, color=COLOR[3], linestyle='-.', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if vel_mode_candidates:
        uniq_h.append(Line2D([0], [0], color=COLOR[3], linestyle='-.', linewidth=0.5))
        uniq_l.append("Fine mode starts")
    if uniq_h:
        ax.legend(uniq_h, uniq_l, fontsize=6)

    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("V", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f"Velocity vs X plot for {title or metric_token}", fontsize=9)
    plt.tight_layout()
    plt.xlim(0, 0.05)
    plt.savefig(f"vel_{metric_token}.png")
    plt.show()
 
def read_final_time(filename):
    # Final time is the last timestamp before "return to max" focus mode
    TIME_COL = 0
    baseline = None
    final_t = None
    raw_vals = []

    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # skip empty/header rows
            if not row:
                continue
            t_raw = safe_float(row[TIME_COL].strip())
            raw_vals.append(t_raw)
            if 'return to max' in row[20].lower():
                break

    if not raw_vals:
        return 0.0

    # detect ns vs s
    convert_ns = any(v is not None and not math.isnan(v) and abs(v) > 1e12 for v in raw_vals)
    # find first and last valid
    valid = [v for v in raw_vals if v is not None and not math.isnan(v)]
    if not valid:
        return 0.0
    first = valid[0]
    last = valid[-1]
    if convert_ns:
        return (last - first) * 1e-9
    else:
        return (last - first)

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
            time, dema_fv, dfv, ddfv, ratio, velocity, x, modes, stop_flag = read_csv_focus_data(csv_path, offset=0.043)
            metric_name = clean_metric_name(d)
            # only add if we actually read some data
            if dema_fv:
                steel_data[metric_name] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
                    "ratio": ratio,
                    "velocity": velocity,
                    "x": x
                    ,"mode": modes
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
            time, dema_fv, dfv, ddfv, ratio, velocity, x, modes, stop_flag = read_csv_focus_data(csv_path, offset=0.043)
            if dema_fv:
                all_data[os.path.basename(d)] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
                    "ratio": ratio,
                    "velocity": velocity,
                    "x": x
                    ,"mode": modes
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
            time, dema_fv, dfv, ddfv, ratio, velocity, x, modes, stop_flag = read_csv_focus_data(csv_path, offset=0)
            if dema_fv:
                all_data[os.path.basename(d)] = {
                    "time": time,
                    "dema_fv": dema_fv,
                    "dfv": dfv,
                    "ddfv": ddfv,
                    "ratio": ratio,
                    "velocity": velocity,
                    "x": x
                    ,"mode": modes
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
    
    # steel_data_adaptive = {k: v for k, v in all_data.items() if 'Steel' in k}
    # plot_1_obj(steel_data_adaptive, "Steel Using Adaptive")
    
    plot_1_metric(all_data, "fswm", title="FSWM")
    plot_1_metric(all_data, "sobel", title="Sobel")
    plot_1_metric(all_data, "squared_gradient", title="Squared Gradient")


if __name__ == "__main__":
    main()