"""
網格設定：
- 左下角： lon0 = 120.0000, lat0 = 21.8800
- 間距： dlon = dlat = 0.03 度
- 網格數： 67 x 120
輸出: (lon, lat, value)
"""

import json
import csv
import argparse
from pathlib import Path

# 已知格網參數
LON0, LAT0 = 120.0, 21.88
DLON, DLAT = 0.03, 0.03
NCOLS, NROWS = 67, 120
NEG_MISS = -999.0

def main():
    parser = argparse.ArgumentParser(description="Extract non-missing grid values (lon,lat,value).")
    parser.add_argument("input", type=Path, help=r"D:\NYCU 2527\Master program\11401\Machine Learning\ML_氣象資料\data.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help=r"D:\NYCU 2527\Master program\11401\Machine Learning\ML_氣象資料")
    args = parser.parse_args()

    # 讀 JSON
    obj = json.loads(args.input.read_text(encoding="utf-8"))
    content = obj["cwaopendata"]["dataset"]["Resource"]["Content"]

    # 切成 float 值
    tokens = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
    values = [float(t.replace("D", "E")) for t in tokens]

    out_path = args.output or args.input.with_suffix(".csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "value"])
        for k, v in enumerate(values):
            if abs(v - NEG_MISS) < 1e-9:
                continue  # 跳過 -999
            row = k // NCOLS
            col = k % NCOLS
            lon = LON0 + col * DLON
            lat = LAT0 + row * DLAT
            w.writerow([f"{lon:.4f}", f"{lat:.4f}", f"{v:.2f}"])  # value 保留兩位小數

    print(f"Done: {out_path}")

if __name__ == "__main__":
    main()
