import os
import csv
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import matplotlib as mpl
import shutil

use_usetex = False
if shutil.which('latex') or shutil.which('pdflatex'):
    use_usetex = True
mpl.rcParams['text.usetex'] = use_usetex
# if not use_usetex:
#     print('Note: LaTeX not found on PATH; using matplotlib mathtext (text.usetex=False).')
# Plotting 3 focus metrics together, first as focus value, then as ratio
ROOT = os.path.dirname(__file__)
PREFIX = "Steel_ehc"
# Color scheme used:
# COLOR = ["#1B9E77", "#D95F02","#7570B3","#CC79A7"] # og color scheme
COLOR = ["#C7495A", "#8DA55F","#4A6C78","#8E8E8E"]
LINESTYLE = ['--', ':', '-','-.']

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

            # Convert to float with safe handling
            dfv_v = safe_float(row[9].strip())
            ddfv_v = safe_float(row[10].strip())
            ratio_v = safe_float(row[11].strip())
            velocity_v = safe_float(row[19].strip())
            t_raw = safe_float(row[0].strip())

            # Read world-frame position
            x_val = safe_float(row[12].strip())
            y_val = safe_float(row[13].strip())
            z_val = safe_float(row[14].strip())

            # Read quaternion
            qx = safe_float(row[15].strip())
            qy = safe_float(row[16].strip())
            qz = safe_float(row[17].strip())
            qw = safe_float(row[18].strip())

            # Normalize quaternion and rotate position into robot body frame.
            # Then compute body-frame XY magnitude (recommended for drift handling).
            qnorm = math.hypot(qx, qy, qz, qw)
            if qnorm == 0:
                # quaternion invalid: fallback to horizontal magnitude in world frame
                x_raw_v = math.hypot(x_val, y_val)
            else:
                qx_u, qy_u, qz_u, qw_u = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm

                # quaternion multiply for (x,y,z,w) tuples
                def quat_mult(a, b):
                    ax, ay, az, aw = a
                    bx, by, bz, bw = b
                    return (
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw,
                        aw*bw - ax*bx - ay*by - az*bz,
                    )

                # rotate vector r by q: project world position to XY plane
                # (drop Z) then rotate into body frame: r_body = q * (r_xy,0) * q_conj
                vq = (x_val, y_val, 0.0, 0.0)
                q = (qx_u, qy_u, qz_u, qw_u)
                q_conj = (-qx_u, -qy_u, -qz_u, qw_u)

                tmp = quat_mult(q, vq)

                # rx is projection onto body-frame forward axis
                rx, ry, rz, _ = quat_mult(tmp, q_conj)
                x_raw_v = rx

            # Append to csv
            dema.append(fv)
            dfv.append(dfv_v if dfv_v is not None else float('nan'))
            ddfv.append(ddfv_v if ddfv_v is not None else float('nan'))
            ratio.append(ratio_v if ratio_v is not None else float('nan'))
            velocity.append(velocity_v if velocity_v is not None else float('nan'))
            times_raw.append(t_raw if t_raw is not None else float('nan'))
            x_raw.append(x_raw_v if x_raw_v is not None else float('nan'))
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
        # shift each metric so its FV peak maps to x=0.025 (consistent with other plots)
        x_plot = compute_shifted_x(metric_data, target_x=0.025)
        ax.plot(
            x_plot,
            metric_data["dema_fv"],
            label=metric_name,
            color=COLOR[i],
            linestyle=LINESTYLE[i],
            linewidth=1
        )

    ax.set_xlabel("$X$ (m)", fontsize=9)
    ax.set_ylabel("$FV$", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    try:
        ax.yaxis.set_offset_position('right')
    except Exception:
        pass
    try:
        # draw to ensure offset text is computed
        fig.canvas.draw()
    except Exception:
        pass
    offset_text = ax.yaxis.get_offset_text().get_text()
    if offset_text:
        # hide the default offset text
        ax.yaxis.get_offset_text().set_visible(False)
        try:
            ticklabels = ax.yaxis.get_ticklabels()
            fontsize = ticklabels[0].get_fontsize() if ticklabels else 8
        except Exception:
            fontsize = 8
        # reduce size slightly so it doesn't dominate the plot
        fontsize_offset = max(6, int(fontsize * 0.85))
        try:
            bbox = ax.get_position()
            # place the offset directly above the y-axis (left or right)
            try:
                side = ax.yaxis.get_offset_position()
            except Exception:
                side = 'left'
            fx = bbox.x1 if side == 'right' else bbox.x0
            fy = bbox.y1 + 0.01
            fig.text(fx, fy, offset_text, ha='center', va='bottom', fontsize=fontsize_offset)
        except Exception:
            # fallback to inside-axes placement
            ax.text(0.01, 0.98, offset_text, transform=ax.transAxes, ha='left', va='top', fontsize=fontsize_offset)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("$FV$ Across Metrics vs Position $X$", fontsize=9)
    # focus the view around the aligned peak (match fv_triplet range)
    ax.set_xlim(0.02, 0.03)
    plt.tight_layout()
    plt.savefig("fv_comparison_vibrant.png", dpi=300, bbox_inches="tight")
    plt.show()
    # plt.close()

    # Also save a log-scale version of the same FV comparison
    try:
        fig_log, ax_log = plt.subplots(figsize=(3.5, 2.8), dpi=300)
        for i, (metric_name, metric_data) in enumerate(steel_data.items()):
            x_plot = compute_shifted_x(metric_data, target_x=0.025)
            ax_log.plot(
                x_plot,
                metric_data["dema_fv"],
                label=metric_name,
                color=COLOR[i],
                linestyle=LINESTYLE[i],
                linewidth=1,
            )

        ax_log.set_xlabel("$X$ (m)", fontsize=9)
        ax_log.set_ylabel("$FV$", fontsize=9)
        ax_log.legend(fontsize=7, loc='upper right')

        # set Y to log scale
        try:
            ax_log.set_yscale('log')
        except Exception:
            pass

        # Format Y tick labels as math-text powers of ten (e.g., $10^{5}$)
        try:
            from matplotlib.ticker import LogFormatterMathtext
            ax_log.yaxis.set_major_formatter(LogFormatterMathtext())
        except Exception:
            pass
        try:
            fig_log.canvas.draw()
        except Exception:
            pass
        offset_text = ax_log.yaxis.get_offset_text().get_text()
        if offset_text:
            ax_log.yaxis.get_offset_text().set_visible(False)
            try:
                ticklabels = ax_log.yaxis.get_ticklabels()
                fontsize = ticklabels[0].get_fontsize() if ticklabels else 8
            except Exception:
                fontsize = 8
            fontsize_offset = max(6, int(fontsize * 0.85))
            try:
                bbox = ax_log.get_position()
                try:
                    side = ax_log.yaxis.get_offset_position()
                except Exception:
                    side = 'left'
                fx = bbox.x1 if side == 'right' else bbox.x0
                fy = bbox.y1 + 0.01
                fig_log.text(fx, fy, offset_text, ha='center', va='bottom', fontsize=fontsize_offset)
            except Exception:
                ax_log.text(0.01, 0.98, offset_text, transform=ax_log.transAxes, ha='left', va='top', fontsize=fontsize_offset)

        ax_log.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # Let matplotlib choose y tick locations automatically for the log plot
        ax_log.tick_params(axis='x', labelsize=8)
        ax_log.tick_params(axis='y', labelsize=8)
        ax_log.set_title("$FV$ Across Metrics vs Position $X$ (log scale)", fontsize=9)
        ax_log.set_xlim(0.01, 0.04)
        plt.tight_layout()
        plt.savefig("fv_comparison_log.png", dpi=300, bbox_inches="tight")
        plt.close(fig_log)
    except Exception:
        pass

    # Plot Smoothed Ratio vs X for all metrics. Simple Moving Average works fine. Don't use EMA since SMA is better for noise and smoothing.
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)    
    for i, (metric_name, metric_data) in enumerate(steel_data.items()):
        x_vals = compute_shifted_x(metric_data, target_x=0.025)
        ratio_vals = metric_data["ratio"]
        # compute and plot moving average
        smoothed = moving_average(ratio_vals, window=11)
        ax.plot(x_vals, smoothed, linewidth=1, label=f"{metric_name}", color=COLOR[i], linestyle=LINESTYLE[i])

    ax.set_xlabel("$X$ (m)", fontsize=9)
    ax.set_ylabel("$Ratio$", fontsize=9)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("$Ratio$ Across Metrics vs Position $X$", fontsize=9)
    # focus the view around the aligned peak (match fv_triplet range)
    ax.set_xlim(0.01, 0.04)
    plt.tight_layout()
    plt.savefig("ratio_comparison.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

def plot_dfv_ddfv(data):
    if not data:
        print("No data to plot.")
        return

    x = compute_shifted_x(data, target_x=0.025)
    dfv = data["dfv"]
    ddfv = data["ddfv"]
    ratio = data["ratio"]
    dema = data["dema_fv"]
    # Switch to horizontal 1x4 layout (preserve same signals and labels)
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.8), dpi=300, sharex=True)

    # Use scientific formatter consistent with fv_comparison_vibrant
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))

    xlim = (0.02, 0.03)
    x_nbins = 3
    y_nbins = 4

    # FV
    ax[0].plot(x, dema, linewidth=1, color=COLOR[2])
    ax[0].set_xlim(*xlim)
    ax[0].set_ylabel("$FV$", fontsize=9)
    ax[0].set_xlabel("$X$ (m)", fontsize=9)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[0].yaxis.set_major_formatter(fmt)

    # dFV
    smoothed_dfv = moving_average(dfv, window=5)
    ax[1].plot(x, smoothed_dfv, linewidth=1, color=COLOR[2])
    ax[1].set_xlim(*xlim)
    ax[1].set_ylabel("\u2207$FV$", fontsize=9)
    ax[1].set_xlabel("$X$ (m)", fontsize=9)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[1].yaxis.set_major_formatter(fmt)

    # ddfv
    smoothed_ddfv = moving_average(ddfv, window=5)
    ax[2].plot(x, smoothed_ddfv, linewidth=1, color=COLOR[2])
    ax[2].set_xlim(*xlim)
    ax[2].set_ylabel("\u2207\u00B2$FV$", fontsize=9)
    ax[2].set_xlabel("$X$ (m)", fontsize=9)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[2].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[2].yaxis.set_major_formatter(fmt)

    # Ratio
    smoothed_ratio = moving_average(ratio, window=11)
    ax[3].plot(x, smoothed_ratio, linewidth=1, color=COLOR[2])
    ax[3].set_xlim(*xlim)
    ax[3].set_xlabel("$X$ (m)", fontsize=9); ax[3].set_ylabel("$Ratio$", fontsize=9)
    ax[3].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[3].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[3].yaxis.set_major_formatter(fmt)

    # For each axis, draw to compute offset text and place it above the axis
    for a in ax:
        try:
            a.yaxis.set_offset_position('right')
        except Exception:
            pass
    try:
        fig.canvas.draw()
    except Exception:
        pass

    for a in ax:
        try:
            off = a.yaxis.get_offset_text().get_text()
            if off:
                a.yaxis.get_offset_text().set_visible(False)
                try:
                    fs = a.yaxis.get_ticklabels()[0].get_fontsize()
                except Exception:
                    fs = 8
                fo = max(6, int(fs * 0.85))
                try:
                    bb = a.get_position()
                    side = 'left'
                    try:
                        side = a.yaxis.get_offset_position()
                    except Exception:
                        pass
                    fx = bb.x1 if side == 'right' else bb.x0
                    fy = bb.y1 + 0.005
                    fig.text(fx, fy, off, ha='center', va='bottom', fontsize=fo)
                except Exception:
                    a.text(0.01, 0.98, off, transform=a.transAxes, ha='left', va='top', fontsize=fo)
        except Exception:
            pass

    # ticks
    for a in ax:
        a.tick_params(axis='x', labelsize=8)
        a.tick_params(axis='y', labelsize=8)

    # suptitle centered and reduced gap
    fig.suptitle('$FV$, \u2207$FV$, \u2207\u00B2$FV$, $Ratio$ vs Position $X$', fontsize=9, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("FV_dFV_ddFV.png", dpi=300, bbox_inches="tight")
    plt.close()

def read_csv_fv_triplet(filename, offset=0):
    focus_vals, ema_vals, dema_vals = [], [], []
    x_raw = []

    def safe_val(v):
        return 0 if v is None or (isinstance(v, float) and math.isnan(v)) else v

    with open(filename, newline='', encoding='utf-8') as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            # stop early if indicated
            try:
                if 'return to max' in row[20].lower():
                    break
            except Exception:
                pass

            # read columns with defensive indexing; skip rows with no focus value
            try:
                focus_v = safe_float(row[6].strip())
            except Exception:
                focus_v = None
            if focus_v is None:
                continue
            try:
                ema_v = safe_float(row[7].strip())
            except Exception:
                ema_v = None
            try:
                dema_v = safe_float(row[8].strip())
            except Exception:
                dema_v = None

            # read x components and make a Euclidean norm like the other reader
            try:
                x_val = safe_float(row[12].strip())
                y_val = safe_float(row[13].strip())
                z_val = safe_float(row[14].strip())
                x_safe = safe_val(x_val)
                y_safe = safe_val(y_val)
                z_safe = safe_val(z_val)
                x_raw_v = math.sqrt(x_safe**2 + y_safe**2 + z_safe**2)
            except Exception:
                x_raw_v = float('nan')

            focus_vals.append(focus_v)
            ema_vals.append(ema_v if ema_v is not None else float('nan'))
            dema_vals.append(dema_v if dema_v is not None else float('nan'))
            x_raw.append(x_raw_v)

    # same x shifting logic as read_csv_focus_data
    first_x = next((v for v in x_raw if not (v is None or math.isnan(v))), None)
    if first_x is None:
        x_vals = [float('nan')] * len(x_raw)
    else:
        shifted = [ (v - first_x) if not (v is None or math.isnan(v)) else float('nan') for v in x_raw ]
        last_valid = next((v for v in reversed(shifted) if not (v is None or math.isnan(v))), None)
        if last_valid is not None and last_valid < 0:
            x_vals = [(-v if not (v is None or math.isnan(v)) else float('nan')) for v in shifted]
        else:
            x_vals = shifted

    x_vals = [ (xx - offset) if not (xx is None or math.isnan(xx)) else float('nan') for xx in x_vals ]

    return x_vals, focus_vals, ema_vals, dema_vals


def plot_fv_triplet(x_vals, focus_vals, ema_vals, dema_vals, outname="fv_triplet.png"):
    # shift so the peak of dema_fv is at x = 0.025
    try:
        arr = np.array(dema_vals, dtype=float)
        idx = int(np.nanargmax(arr))
        peak_x = float(x_vals[idx])
        shift = 0.025 - peak_x
        x_plot = [(xx + shift) for xx in x_vals]
    except Exception:
        x_plot = x_vals

    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    ax.plot(x_plot, dema_vals, label=r'$\mathrm{FV}$', color=COLOR[0], linestyle='-', linewidth=1)
    ax.plot(x_plot, ema_vals, label=r'$\overline{FV}$', color=COLOR[1], linestyle='--', linewidth=1)
    ax.plot(x_plot, focus_vals, label=r'$FV_{o}$', color=COLOR[2], linestyle=':', linewidth=1)

    ax.set_xlabel("$X$ (m)", fontsize=9)
    ax.set_ylabel("$FV$", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    # Use scientific formatter for Y axis (match style in plot_3_metrics)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    try:
        ax.yaxis.set_offset_position('right')
    except Exception:
        pass
    try:
        # draw to ensure offset text is computed
        fig.canvas.draw()
    except Exception:
        pass
    offset_text = ax.yaxis.get_offset_text().get_text()
    if offset_text:
        ax.yaxis.get_offset_text().set_visible(False)
        try:
            ticklabels = ax.yaxis.get_ticklabels()
            fontsize = ticklabels[0].get_fontsize() if ticklabels else 8
        except Exception:
            fontsize = 8
        fontsize_offset = max(6, int(fontsize * 0.85))
        try:
            bbox = ax.get_position()
            # place the offset directly above the y-axis (left or right)
            try:
                side = ax.yaxis.get_offset_position()
            except Exception:
                side = 'left'
            fx = bbox.x1 if side == 'right' else bbox.x0
            fy = bbox.y1 + 0.01
            fig.text(fx, fy, offset_text, ha='center', va='bottom', fontsize=fontsize_offset)
        except Exception:
            ax.text(0.01, 0.98, offset_text, transform=ax.transAxes, ha='left', va='top', fontsize=fontsize_offset)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title("Raw and smoothed $FV$ vs Position $X$", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.xlim(0.02, 0.03)
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_1_obj(data, dataset_name):
    if not data:
        print("No data to plot.")
        return

    # Plot Focus Value and smoothed Velocity for each metric, shifted so FV peak maps to 0.025
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    for i, (metric_name, metric_data) in enumerate(data.items()):
        label = clean_metric_name(metric_name)
        # fv = metric_data.get("dema_fv", [])
        # shift so peak FV is at 0.025 (match plot_1_metric)
        shifted_x = compute_shifted_x(metric_data, target_x=0.025)

        # smoothed velocity
        vel = metric_data.get("velocity", [])
        smoothed_vel = moving_average(vel, window=17)
        # Prepare plot data: keep values up to the last valid point, then
        # add a vertical drop to 0 and extend flat at 0 until x = 0.05.
        x_vals = list(shifted_x)
        y_vals = list(smoothed_vel)

        # find last valid index where both x and y are finite
        last_idx = None
        for j in range(len(x_vals) - 1, -1, -1):
            xv = x_vals[j]
            yv = y_vals[j]
            try:
                if xv is None or yv is None:
                    continue
                if math.isnan(xv) or math.isnan(yv):
                    continue
            except Exception:
                continue
            last_idx = j
            break

        if last_idx is None:
            # nothing to plot
            x_plot = x_vals
            y_plot = y_vals
        else:
            # trim arrays to the last valid point (drop trailing NaNs)
            x_plot = x_vals[: last_idx + 1]
            y_plot = y_vals[: last_idx + 1]

            last_x = x_plot[-1]
            last_y = y_plot[-1]

            # only extend if last_x is less than the axis limit we use (0.05)
            if last_x < 0.05:
                # if last_y is not already zero, append a point at (last_x, 0)
                try:
                    is_zero = (last_y == 0.0)
                except Exception:
                    is_zero = False

                if not is_zero:
                    x_plot.append(last_x)
                    y_plot.append(0.0)

                # append the final flat point at x=0.05 (zero velocity)
                x_plot.append(0.05)
                y_plot.append(0.0)

        ax.plot(
            x_plot,
            y_plot,
            label=label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )
   

    # ax[0].set_ylabel("Focus Value", fontsize=9)
    ax.set_ylabel("$Velocity$", fontsize=9)
    ax.set_xlabel("$X$ (m)", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    # ax.set_title("Velocity vs Position $X$ for {dataset_name}" if dataset_name else "smth", fontsize=9)
    ax.set_title("$Velocity$ vs Position $X$", fontsize=9)
    plt.tight_layout()
    plt.xlim(0.01, 0.03)
    plt.savefig("smoothed_vel.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

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
    
    # FV vs X (create 1x3 horizontal subplots: FV, Ratio, Velocity)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8), dpi=300, sharex=True)
    # ensure ax is indexable in the same way as before
    
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        fv = metric_data.get("dema_fv", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        ax[0].plot(
            shifted_x,
            fv,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )

    # # draw vertical line at first 'fine' mode from primary run
    # x_mark = None
    # primary = next(iter(selected.values()))
    # modes = primary['mode']
    # shifted_primary_x = compute_shifted_x(primary, target_x=target_x)
    # for idx, m in enumerate(modes):
    #     if m and 'fine' in m.lower():
    #         if idx < len(shifted_primary_x):
    #             x_mark = shifted_primary_x[idx]
    #             ax[0].axvline(x_mark, color=COLOR[3], linestyle=LINESTYLE[3], linewidth=1, label='Fine start')
    #         break

    # dedupe legend just in case
    handles, labels = ax[0].get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax[0].legend(uniq_h, uniq_l, fontsize=7, loc='upper left')

    # ax[0].set_xlabel("$X$ (m)", fontsize=9)

    x_nbins = 4
    y_nbins = 4

    ax[0].set_ylabel("$FV$", fontsize=9)
    ax[0].set_xlabel("$X$ (m)", fontsize=9)

    # Disable scientific notation / offset for the results FV axis so the
    # Y-axis labels appear as plain numbers in `results_{metric}.png`.
    try:
        ax[0].ticklabel_format(style='plain', axis='y', useOffset=False)
        ax[0].yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass

    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[0].tick_params(axis='x', labelsize=8)
    ax[0].tick_params(axis='y', labelsize=8)

    # Ratio vs X
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        ratio = metric_data.get("ratio", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        ax[1].plot(
            shifted_x,
            ratio,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )
    # ax[1].axvline(x_mark, color=COLOR[3], linestyle=LINESTYLE[3], linewidth=1, label='Fine start')

    # ax[1].set_xlabel("$X$ (m)", fontsize=9)
    ax[1].set_ylabel("$Ratio$", fontsize=9)
    ax[1].set_xlabel("$X$ (m)", fontsize=9)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[1].tick_params(axis='x', labelsize=8)
    ax[1].tick_params(axis='y', labelsize=8)

    # Velocity vs X
    # fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    for i, (mat_label, metric_data) in enumerate(selected.items()):
        vel = metric_data.get("velocity", [])
        target_x = 0.025
        shifted_x = compute_shifted_x(metric_data, target_x=target_x)

        # prepare x/y lists and trim trailing NaNs, then extend velocity to zero
        x_vals = list(shifted_x)
        y_vals = list(vel)

        # find last valid index where both x and y are finite
        last_idx = None
        for j in range(len(x_vals) - 1, -1, -1):
            xv = x_vals[j]
            yv = y_vals[j]
            try:
                if xv is None or yv is None:
                    continue
                if math.isnan(xv) or math.isnan(yv):
                    continue
            except Exception:
                continue
            last_idx = j
            break

        if last_idx is None:
            x_plot = x_vals
            y_plot = y_vals
        else:
            x_plot = x_vals[: last_idx + 1]
            y_plot = y_vals[: last_idx + 1]

            last_x = x_plot[-1]
            last_y = y_plot[-1]

            # if the last x is before the axis limit, append a drop to zero and flat to 0.03
            if last_x < 0.03:
                try:
                    is_zero = (last_y == 0.0)
                except Exception:
                    is_zero = False

                if not is_zero:
                    x_plot.append(last_x)
                    y_plot.append(0.0)

                x_plot.append(0.03)
                y_plot.append(0.0)

        ax[2].plot(
            x_plot,
            y_plot,
            label=mat_label,
            color=COLOR[i % len(COLOR)],
            linestyle=LINESTYLE[i % len(LINESTYLE)],
            linewidth=1,
        )
    # ax[2].axvline(x_mark, color=COLOR[3], linestyle=LINESTYLE[3], linewidth=1, label='Fine start')

    ax[2].set_xlabel("$X$ (m)", fontsize=9)
    ax[2].set_ylabel("$Velocity$", fontsize=9)
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
    ax[2].yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    ax[2].tick_params(axis='x', labelsize=8)
    ax[2].tick_params(axis='y', labelsize=8)
    # ax[2].set_title(f"Velocity vs X Plot for {title or metric_token}", fontsize=9)
    # Put the suptitle before tightening layout and reserve top space
    fig.suptitle(f'$FV$, $Ratio$, $Velocity$ vs Position $X$ for {title or metric_token}', fontsize=9, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave top space for suptitle
    plt.xlim(0.02, 0.026)
    # Hardcode x tick marks for this results figure only
    xticks = [0.02, 0.022, 0.024, 0.026]
    try:
        for a in ax:
            a.set_xticks(xticks)
    except Exception:
        # fallback in case ax is not iterable (shouldn't happen here)
        try:
            ax.set_xticks(xticks)
        except Exception:
            pass
    plt.savefig(f"results_{metric_token}.png", dpi=300, bbox_inches="tight")
    plt.close()
 
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

    # Additionally plot raw FV variants (focus_value, ema_fv, dema_fv).
    try:
        # use CF EHC Sobel run (assumed to exist)
        steel_fswm_dir = next((d for d in find_alg_dirs(ROOT, 'steel') if 'ehc' in d.lower() and 'fswm' in d.lower()))
        csv_to_use = first_csv_in_dir(steel_fswm_dir)
        if csv_to_use:
            x_vals, focus_vals, ema_vals, dema_vals = read_csv_fv_triplet(csv_to_use, offset=0.043)
            plot_fv_triplet(x_vals, focus_vals, ema_vals, dema_vals, outname="fv_triplet.png")
    except Exception:
        # deliberately not falling back to steel; surface error silently
        pass

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
                # compute max ratio safely (ignore None/NaN)
                valid_ratios = [r for r in ratio if r is not None and not (isinstance(r, float) and math.isnan(r))]
                if valid_ratios:
                    max_ratio = max(valid_ratios)
                else:
                    max_ratio = float('nan')

                # write both max FV location and the max values
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
                # compute max ratio safely (ignore None/NaN)
                valid_ratios = [r for r in ratio if r is not None and not (isinstance(r, float) and math.isnan(r))]
                if valid_ratios:
                    max_ratio = max(valid_ratios)
                else:
                    max_ratio = float('nan')

                write_to_csv('Max_Focus_X.csv', os.path.basename(d), max_focus_x, max_focus)
                write_to_csv('Max_FV_Ratio.csv', os.path.basename(d), max_focus, max_ratio)
            else:
                print(f"No valid data read from ({os.path.basename(csv_path)}).")
            if stop_flag:
                continue
        else:
            print(f"No .csv file found in {d}")
    
    steel_data_adaptive = {k: v for k, v in all_data.items() if 'Steel' in k}
    plot_1_obj(steel_data_adaptive, "Steel Adaptive")
    
    plot_1_metric(all_data, "fswm", title="FSWM")
    plot_1_metric(all_data, "sobel", title="Sobel")
    plot_1_metric(all_data, "squared_gradient", title="Squared Gradient")


if __name__ == "__main__":
    main()