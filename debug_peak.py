import os
import numpy as np
import plotting

ROOT = os.path.dirname(plotting.__file__)
print('ROOT', ROOT)
steel_dirs = plotting.find_steel_ehc_dirs(ROOT)
print('Found steel dirs:', len(steel_dirs))
for d in steel_dirs:
    csv = plotting.first_csv_in_dir(d)
    if not csv:
        continue
    time,dema,dfv,ddfv,ratio,velocity,x,modes,stop = plotting.read_csv_focus_data(csv, offset=0.043)
    if dema:
        arr = np.array(dema, dtype=float)
        idx = int(np.nanargmax(arr))
        print(os.path.basename(d), 'peak_idx=', idx, 'peak_x=', x[idx])
        # show peak after compute_shifted_x
        shifted = plotting.compute_shifted_x({"x": x, "dema_fv": dema}, target_x=0.025)
        try:
            print('  shifted_peak_x =', shifted[idx])
        except Exception:
            print('  shifted_peak_x: could not compute')

# Also check fv_triplet source file
try:
    steel_fswm_dir = next((d for d in plotting.find_alg_dirs(ROOT, 'steel') if 'ehc' in d.lower() and 'fswm' in d.lower()))
    csv = plotting.first_csv_in_dir(steel_fswm_dir)
    if csv:
        x_vals, focus_vals, ema_vals, dema_vals = plotting.read_csv_fv_triplet(csv, offset=0.043)
        arr = np.array(dema_vals, dtype=float)
        idx = int(np.nanargmax(arr))
        print('fv_triplet source', os.path.basename(steel_fswm_dir), 'peak_idx=', idx, 'peak_x=', x_vals[idx])
except StopIteration:
    pass

print('Done')
