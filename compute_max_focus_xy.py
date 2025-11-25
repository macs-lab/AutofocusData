import os
import csv
import math
import numpy as np

ROOT = os.path.dirname(__file__)
OUTFILE = os.path.join(ROOT, '..', 'Max_Focus_X.csv')

# Helper: find first CSV in a directory
def first_csv_in_dir(d):
    try:
        for name in sorted(os.listdir(d)):
            if name.lower().endswith('.csv'):
                return os.path.join(d, name)
    except Exception:
        return None
    return None

# Safe float
def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None

# Read CSV and return arrays
def read_csv_arrays(path):
    dema = []
    xs = []
    ys = []
    zs = []
    ox = []
    oy = []
    oz = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                # if row too short skip
                if len(row) <= 8:
                    continue
                fv = safe_float(row[8].strip())
                if fv is None:
                    continue
                dema.append(abs(fv))
                # positions
                x = safe_float(row[12].strip()) if len(row) > 12 else None
                y = safe_float(row[13].strip()) if len(row) > 13 else None
                z = safe_float(row[14].strip()) if len(row) > 14 else None
                xs.append(x if x is not None else float('nan'))
                ys.append(y if y is not None else float('nan'))
                zs.append(z if z is not None else float('nan'))
                # orientation
                ox.append(safe_float(row[16].strip()) if len(row) > 16 else float('nan'))
                oy.append(safe_float(row[17].strip()) if len(row) > 17 else float('nan'))
                oz.append(safe_float(row[18].strip()) if len(row) > 18 else float('nan'))
                # stop marker
                try:
                    if 'return to max' in row[20].lower():
                        break
                except Exception:
                    pass
    except Exception as e:
        print('Failed reading', path, e)
    return dema, xs, ys, zs, ox, oy, oz

# Compute projected XY at index
def compute_projected_xy(x, y, oz):
    # prefer orientation yaw (oz). If missing, fallback to sqrt(x^2+y^2)
    try:
        if oz is not None and not (isinstance(oz, float) and math.isnan(oz)):
            yaw = oz
            # detect degrees (> 2*pi) and convert
            if abs(yaw) > 2 * math.pi:
                yaw = math.radians(yaw)
            cx = math.cos(yaw)
            sy = math.sin(yaw)
            if x is None or y is None or math.isnan(x) or math.isnan(y):
                return float('nan')
            return x * cx + y * sy
    except Exception:
        pass
    # fallback
    try:
        if x is None or y is None:
            return float('nan')
        return math.sqrt(x * x + y * y)
    except Exception:
        return float('nan')

# Walk subdirectories in ROOT and process each folder that contains a csv
runs = []
for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    if os.path.isdir(d):
        csvp = first_csv_in_dir(d)
        if csvp:
            runs.append((name, csvp))

# write header (overwrite)
with open(OUTFILE, 'w', newline='', encoding='utf-8') as out:
    w = csv.writer(out)
    w.writerow(['run', 'projected_xy_m', 'max_fv'])

    for run_name, csvp in runs:
        dema, xs, ys, zs, ox, oy, oz = read_csv_arrays(csvp)
        if not dema:
            print('no fv for', run_name)
            continue
        arr = np.array(dema, dtype=float)
        if np.all(np.isnan(arr)):
            print('all nan fv', run_name)
            continue
        try:
            idx = int(np.nanargmax(arr))
        except Exception:
            # fallback
            best = float('-inf')
            idx = None
            for i, v in enumerate(dema):
                try:
                    if v is None or math.isnan(v):
                        continue
                except Exception:
                    continue
                if v > best:
                    best = v
                    idx = i
            if idx is None:
                print('no idx', run_name)
                continue
        # gather values at idx
        try:
            x = xs[idx]
        except Exception:
            x = float('nan')
        try:
            y = ys[idx]
        except Exception:
            y = float('nan')
        try:
            oz_v = oz[idx]
        except Exception:
            oz_v = float('nan')
        projected = compute_projected_xy(x, y, oz_v)
        maxfv = float(arr[idx]) if not math.isnan(arr[idx]) else float('nan')
        w.writerow([run_name, projected, maxfv])
        print('wrote', run_name, projected, maxfv)

print('Wrote', OUTFILE)
