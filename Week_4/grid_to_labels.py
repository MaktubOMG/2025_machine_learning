"""
假設已知網格參數：
- 左下角： lon0=120.0000, lat0=21.8800
- 間距： dlon=dlat=0.03 度
- 大小： ncols=67（經向，西→東），nrows=120（緯向，南→北）
輸出:(lon, lat, label)
"""

import json
import csv
import argparse # argparse套件，會要求在命令列提供必要參數(:python grid_to_labels.py data.json)
from pathlib import Path

# 已知網格參數
LON0 = 120.0000
LAT0 = 21.8800
DLON = 0.03
DLAT = 0.03
NCOLS = 67
NROWS = 120
NEG_MISS = -999.0

def main():
    parser = argparse.ArgumentParser(description="Convert data.json to lon,lat,label (simple, no checks).")
    parser.add_argument("input", type=Path, help=r"D:\NYCU 2527\Master program\11401\Machine Learning\ML_氣象資料\data.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help=r"D:\NYCU 2527\Master program\11401\Machine Learning\ML_氣象資料")
    args = parser.parse_args()

    # 讀 JSON，定位到 Content 字串
    obj = json.loads(args.input.read_text(encoding="utf-8"))
    # 結構：cwaopendata -> dataset -> Resource -> Content
    content = obj["cwaopendata"]["dataset"]["Resource"]["Content"]

    # 切成字串陣列並轉 float（支援科學記號、Fortran 'D'）
    tokens = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip() != ""]
    values = [float(t.replace("D", "E")) for t in tokens]

    # 逐點輸出 lon,lat,label（不做維度檢查）
    out_path = args.output or args.input.with_suffix(".csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "label"])
        for k, v in enumerate(values):
            row = k // NCOLS         # 0..119（南→北）
            col = k % NCOLS          # 0..66  （西→東）
            lon = LON0 + col * DLON
            lat = LAT0 + row * DLAT
            label = 0 if abs(v - NEG_MISS) < 1e-9 else 1
            w.writerow([f"{lon:.4f}", f"{lat:.4f}", label])

    print(f"Done: {out_path}")

if __name__ == "__main__":
    main()
