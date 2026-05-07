"""
================================================================
DATASET RITEL — Struktur Data Python
Proyek: Prediksi & Pencegahan Customer Churn
================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── 1. DEFINISI KOLOM & TIPE DATA ──────────────────────────────

COLUMN_SCHEMA = {
    "No"                       : "int64",
    "Row_Num"                  : "int64",
    "Customer_ID"              : "object",
    "Product"                  : "object",
    "First_Transaction"        : "datetime64[ns]",
    "Last_Transaction"         : "datetime64[ns]",
    "Average_Transaction_Amount": "float64",
    "Count_Transaction"        : "int64",
}


# ── 2. SAMPLE DATA (10 BARIS) ───────────────────────────────────

sample_data = {
    "No": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "Row_Num": [1001, 1002, 1003, 1004, 1005,
                1006, 1007, 1008, 1009, 1010],

    "Customer_ID": [
        "CUST-0001", "CUST-0002", "CUST-0003", "CUST-0004", "CUST-0005",
        "CUST-0006", "CUST-0007", "CUST-0008", "CUST-0009", "CUST-0010",
    ],

    "Product": [
        "Sepatu", "Jaket", "Sepatu", "Tas",    "Baju",
        "Jaket",  "Sepatu","Baju",   "Tas",    "Jaket",
    ],

    "First_Transaction": pd.to_datetime([
        "2019-03-15", "2018-07-22", "2020-01-10", "2017-11-05", "2021-06-30",
        "2018-02-14", "2019-09-01", "2020-05-18", "2016-12-25", "2022-03-08",
    ]),

    "Last_Transaction": pd.to_datetime([
        "2023-08-10", "2022-04-01", "2021-09-15", "2023-11-20", "2023-12-01",
        "2021-01-30", "2022-07-07", "2020-11-11", "2023-10-05", "2023-06-15",
    ]),

    "Average_Transaction_Amount": [
        450_000, 850_000, 320_000, 670_000, 210_000,
        920_000, 380_000, 150_000, 730_000, 560_000,
    ],

    "Count_Transaction": [12, 7, 4, 18, 3, 9, 6, 2, 21, 5],
}


# ── 3. BUAT DATAFRAME ───────────────────────────────────────────

df = pd.DataFrame(sample_data)

# Terapkan tipe data sesuai skema
for col, dtype in COLUMN_SCHEMA.items():
    if "datetime" not in dtype:
        df[col] = df[col].astype(dtype)


# ── 4. FEATURE ENGINEERING ─────────────────────────────────────

CHURN_THRESHOLD_DAYS = 180   # 6 bulan inaktivitas = churn
REFERENCE_DATE       = datetime(2024, 1, 1)

df["Year_First_Transaction"] = df["First_Transaction"].dt.year
df["Year_Last_Transaction"]  = df["Last_Transaction"].dt.year
df["Days_Since_Last_Trans"]  = (
    REFERENCE_DATE - df["Last_Transaction"]
).dt.days
df["Tenure_Days"] = (
    df["Last_Transaction"] - df["First_Transaction"]
).dt.days

# Label churn: 1 = Churn, 0 = Aktif
df["Is_Churn"] = (
    df["Days_Since_Last_Trans"] >= CHURN_THRESHOLD_DAYS
).astype(int)


# ── 5. RINGKASAN DATASET ────────────────────────────────────────

def tampilkan_ringkasan(dataframe: pd.DataFrame) -> None:
    """Cetak ringkasan statistik dan informasi dataset."""

    print("=" * 60)
    print("  DATASET RITEL — RINGKASAN")
    print("=" * 60)

    print(f"\n  Jumlah baris    : {dataframe.shape[0]:,}")
    print(f"  Jumlah kolom    : {dataframe.shape[1]}")

    total     = len(dataframe)
    n_churn   = dataframe["Is_Churn"].sum()
    n_aktif   = total - n_churn

    print(f"\n  Pelanggan Churn : {n_churn:,} ({n_churn/total*100:.1f}%)")
    print(f"  Pelanggan Aktif : {n_aktif:,} ({n_aktif/total*100:.1f}%)")

    print("\n  Tipe Data Kolom:")
    print("-" * 60)
    for col, dtype in dataframe.dtypes.items():
        print(f"  {col:<35} {str(dtype)}")

    print("\n  Statistik Numerik:")
    print("-" * 60)
    print(
        dataframe[[
            "Average_Transaction_Amount",
            "Count_Transaction",
            "Days_Since_Last_Trans",
            "Tenure_Days",
        ]].describe().round(2).to_string()
    )

    print("\n  Missing Values:")
    print("-" * 60)
    mv = dataframe.isnull().sum()
    if mv.sum() == 0:
        print("  Tidak ada missing values.")
    else:
        print(mv[mv > 0])

    print("\n  5 Baris Pertama:")
    print("-" * 60)
    print(dataframe.head().to_string(index=False))
    print("=" * 60)


# ── 6. FUNGSI VALIDASI DATA ─────────────────────────────────────

def validasi_data(dataframe: pd.DataFrame) -> dict:
    """
    Validasi integritas data sebelum digunakan untuk modeling.
    Mengembalikan dictionary berisi hasil validasi.
    """
    hasil = {}

    # Cek duplikasi Customer_ID
    hasil["duplikasi_customer"] = dataframe["Customer_ID"].duplicated().sum()

    # Cek nilai negatif pada kolom numerik
    hasil["transaksi_negatif"] = (
        dataframe["Average_Transaction_Amount"] < 0
    ).sum()

    # Cek logika tanggal (First > Last adalah error)
    hasil["tanggal_tidak_valid"] = (
        dataframe["First_Transaction"] > dataframe["Last_Transaction"]
    ).sum()

    # Cek Count_Transaction = 0 (anomali)
    hasil["count_nol"] = (dataframe["Count_Transaction"] == 0).sum()

    print("\n  Hasil Validasi Data:")
    print("-" * 60)
    for k, v in hasil.items():
        status = "LULUS" if v == 0 else f"PERLU DICEK ({v} baris)"
        print(f"  {k:<35} {status}")

    return hasil


# ── 7. JALANKAN ─────────────────────────────────────────────────

if __name__ == "__main__":
    tampilkan_ringkasan(df)
    validasi_data(df)

    # Ekspor ke CSV
    df.to_csv("retail_dataset.csv", index=False)
    print("\n  Dataset berhasil diekspor: retail_dataset.csv")
