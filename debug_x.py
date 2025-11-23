import csv, math
csvf = r'c:\Users\ayaha\Desktop\adaptive_data\Fall 25\AutofocusData\Steel_ehc_sobel_1_1761332318\Steel_ehc_sobel_1_1761332318.csv'
with open(csvf, newline='', encoding='utf-8') as f:
    r = csv.reader(f, delimiter=',')
    rows = []
    for i, row in enumerate(r):
        if i > 30:
            break
        rows.append(row)

for i, row in enumerate(rows[:15]):
    try:
        x_val = float(row[12].strip())
        y_val = float(row[13].strip())
        z_val = float(row[14].strip())
        norm = math.sqrt(x_val**2 + y_val**2 + z_val**2)
    except Exception:
        norm = 'NA'
    print(i, 'raw cols12-14:', row[12:15], 'norm:', norm)

# Also print the FV columns (0-based): indices 6,7,8
for i, row in enumerate(rows[:15]):
    vals = []
    for idx in (6,7,8):
        try:
            vals.append(row[idx])
        except Exception:
            vals.append('NA')
    print(i, 'cols 6-8:', vals)
