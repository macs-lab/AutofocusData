import os
import csv
from statistics import median
from collections import defaultdict

INPUT_PATH = "C:/Users/ayaha/Desktop/adaptive_data/Fall 25/Time_Taken.csv"
BACKUP_PATH = "C:/Users/ayaha/Desktop/adaptive_data/Fall 25/Time_Taken_original.csv"
TARGET_OBJECTS = {"CF", "PCB", "Steel"}

# Splits and extracts CF, PCB, Steel (that were taken at a slightly further distance than the others)
# Compute the median time for EHC and adaptive modes for all objects
# Offset = object_mode_mean - mode_median
# New value = original value - offset
# Time is then new_time = original_time - offset

def parse_row(line):
    # Expect format: name,value,maybe-empty
    parts = line.strip().split(',')
    if len(parts) < 2 or parts[0] == '':
        return None
    name = parts[0]
    try:
        value = float(parts[1])
    except Exception:
        return None
    tokens = name.split('_')
    # object is tokens[0], mode commonly tokens[1]
    obj = tokens[0]
    mode = tokens[1] if len(tokens) > 1 else 'unknown'
    return name, obj, mode, value


def load_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parsed = parse_row(line)
        if parsed:
            rows.append(parsed)
    return rows, lines


def write_backup(orig_lines, backup_path):
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(orig_lines)


def write_rows(rows, out_path, orig_lines):
    # We'll reconstruct file preserving lines for lines we couldn't parse
    out_lines = []
    # Build a map name->newvalue
    newvals = {r[0]: r[3] for r in rows}
    for line in orig_lines:
        parsed = parse_row(line)
        if parsed and parsed[0] in newvals:
            out_lines.append(f"{parsed[0]},{newvals[parsed[0]]},\n")
        else:
            out_lines.append(line)
    # Write to a temp file first to avoid permission issues; then try to move into place.
    tmp_path = out_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        # If replace fails (file locked), keep tmp and raise so user can replace manually.
        print(f"Could not replace {out_path} (maybe it's open). Kept {tmp_path} instead.")
        raise


def main():
    # If a backup exists, use it as the source so we don't repeatedly compound edits.
    source_path = BACKUP_PATH if os.path.exists(BACKUP_PATH) else INPUT_PATH
    rows, orig_lines = load_rows(source_path)
    if not rows:
        print("No parsable rows found in", source_path)
        return

    # Group by mode
    mode_times = defaultdict(list)
    for name, obj, mode, val in rows:
        mode_times[mode].append(val)

    mode_medians = {m: median(v) for m, v in mode_times.items()}
    print("Mode medians:")
    for m, med in mode_medians.items():
        print(f"  {m}: {med:.4f}")

    # Compute per-target offsets per mode
    # For each target object and mode, compute mean of its rows and offset = mean - mode_median
    # Compute per-target offsets per mode
    # For each target object and mode, compute mean of its rows and offset = mean - mode_median
    # Special case: treat CF's 'adaptive' rows as if they compare to the 'default' median
    offsets = defaultdict(dict)  # obj -> mode -> offset
    for obj in TARGET_OBJECTS:
        for m in mode_medians:
            vals = [val for name,o,mode,val in rows if o==obj and mode==m]
            if vals:
                mean_val = sum(vals)/len(vals)
                # use default median for CF adaptive per user request
                if obj == 'CF' and m == 'adaptive' and 'default' in mode_medians:
                    compare_med = mode_medians['default']
                else:
                    compare_med = mode_medians[m]
                offsets[obj][m] = mean_val - compare_med

    print("Computed offsets (to subtract):")
    for obj, md in offsets.items():
        for m, off in md.items():
            print(f"  {obj} {m}: {off:.4f}")

    # Apply trimming: subtract offset from each matching row
    new_rows = []
    for name, obj, mode, val in rows:
        newval = val
        if obj in offsets and mode in offsets[obj]:
            off = offsets[obj][mode]
            newval = val - off
            if newval < 0:
                newval = 0.0
        new_rows.append((name, obj, mode, newval))

    # Backup original
    write_backup(orig_lines, BACKUP_PATH)
    # Write updated file
    write_rows(new_rows, INPUT_PATH, orig_lines)

    print(f"Backup written to {BACKUP_PATH}")
    print(f"Updated {INPUT_PATH} with trimmed values.")

if __name__ == '__main__':
    main()
