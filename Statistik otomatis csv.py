"""
statistik_otomatis_csv.py
Bonus – Program Otomatis Statistik dari File CSV
Mata Kuliah: Statistika dan Probabilitas

Cara Penggunaan:
  python statistik_otomatis_csv.py data_nilai.csv --mu0 75 --sigma 40 --alpha 0.05

Format CSV:
  nilai
  78
  80
  75
  ...
"""

import csv
import sys
import math
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats


# ============================================================
# FUNGSI BANTU
# ============================================================

def baca_csv(filepath: str, kolom: str = 'nilai') -> list:
    """Membaca data numerik dari file CSV."""
    data = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append(float(row[kolom]))
            except (ValueError, KeyError):
                pass
    if not data:
        raise ValueError(f"Tidak ada data numerik pada kolom '{kolom}' di file {filepath}.")
    return data


def hitung_mean(data: list) -> float:
    return sum(data) / len(data)


def hitung_std(data: list) -> float:
    n = len(data)
    mean = hitung_mean(data)
    return math.sqrt(sum((xi - mean) ** 2 for xi in data) / (n - 1))


def hitung_ci(data, alpha=0.05):
    n = len(data)
    mean = hitung_mean(data)
    s = hitung_std(data)
    se = s / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    me = t_crit * se
    return mean, s, se, t_crit, me, mean - me, mean + me


def uji_z(data, mu_0, sigma, alpha=0.05):
    n = len(data)
    x_bar = hitung_mean(data)
    se = sigma / math.sqrt(n)
    z_hit = (x_bar - mu_0) / se
    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_hit)))
    keputusan = "TOLAK H₀" if abs(z_hit) > z_crit else "GAGAL TOLAK H₀"
    return x_bar, se, z_hit, z_crit, p_val, keputusan


def uji_t(data, mu_0, alpha=0.05):
    n = len(data)
    mean = hitung_mean(data)
    s = hitung_std(data)
    se = s / math.sqrt(n)
    t_hit = (mean - mu_0) / se
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    p_val = 2 * (1 - stats.t.cdf(abs(t_hit), df=df))
    keputusan = "TOLAK H₀" if abs(t_hit) > t_crit else "GAGAL TOLAK H₀"
    return mean, s, se, t_hit, df, t_crit, p_val, keputusan


# ============================================================
# FUNGSI VISUALISASI
# ============================================================

def buat_grafik(data, mu_0, sigma, alpha, output_dir):
    n = len(data)

    # Hitung semua nilai
    (mean, s, se_ci, t_crit_ci, me, ci_low, ci_up) = hitung_ci(data, alpha)
    (x_bar, se_z, z_hit, z_crit, p_z, kep_z) = uji_z(data, mu_0, sigma, alpha)
    (_, _, se_t, t_hit, df, t_crit, p_t, kep_t) = uji_t(data, mu_0, alpha)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Laporan Statistik Otomatis dari CSV\nStatistika dan Probabilitas",
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Panel 1: Histogram ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data, bins=max(5, int(n ** 0.5)), color='steelblue',
             edgecolor='white', alpha=0.85)
    ax1.axvline(mean, color='red', lw=2.5, label=f'Mean={mean:.2f}')
    ax1.set_title('Histogram Data', fontweight='bold')
    ax1.set_xlabel('Nilai')
    ax1.set_ylabel('Frekuensi')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- Panel 2: Confidence Interval ----
    ax2 = fig.add_subplot(gs[0, 1])
    x_ci = np.linspace(mean - 4 * se_ci, mean + 4 * se_ci, 500)
    y_ci = stats.norm.pdf(x_ci, loc=mean, scale=se_ci)
    ax2.plot(x_ci, y_ci, 'steelblue', lw=2.5, label='Distribusi Sampling')
    xf = np.linspace(ci_low, ci_up, 300)
    ax2.fill_between(xf, stats.norm.pdf(xf, loc=mean, scale=se_ci),
                     alpha=0.35, color='orange', label=f'CI 95%: [{ci_low:.2f}, {ci_up:.2f}]')
    ax2.axvline(mean, color='red', lw=2, label=f'Mean={mean:.2f}')
    ax2.axvline(ci_low, color='darkorange', lw=1.5, linestyle='--')
    ax2.axvline(ci_up, color='darkgreen', lw=1.5, linestyle='--')
    ax2.set_title('Confidence Interval 95%', fontweight='bold')
    ax2.set_xlabel('Nilai Mean')
    ax2.set_ylabel('Densitas')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ---- Panel 3: Uji Z ----
    ax3 = fig.add_subplot(gs[0, 2])
    xz = np.linspace(-4, 4, 1000)
    yz = stats.norm.pdf(xz)
    ax3.plot(xz, yz, 'steelblue', lw=2.5, label='N(0,1)')
    ax3.fill_between(np.linspace(-4, -z_crit, 300),
                     stats.norm.pdf(np.linspace(-4, -z_crit, 300)),
                     alpha=0.55, color='red', label='Tolak H₀')
    ax3.fill_between(np.linspace(z_crit, 4, 300),
                     stats.norm.pdf(np.linspace(z_crit, 4, 300)),
                     alpha=0.55, color='red')
    ax3.fill_between(np.linspace(-z_crit, z_crit, 500),
                     stats.norm.pdf(np.linspace(-z_crit, z_crit, 500)),
                     alpha=0.25, color='green', label='Terima H₀')
    z_plot = max(min(z_hit, 3.8), -3.8)
    ax3.axvline(z_plot, color='purple', lw=2.5, label=f'Z-hit={z_hit:.4f}')
    ax3.axvline(-z_crit, color='orange', lw=1.8, linestyle='--',
                label=f'Z-krit=±{z_crit:.4f}')
    ax3.axvline(z_crit, color='orange', lw=1.8, linestyle='--')
    ax3.set_title(f'Uji Z – {kep_z}', fontweight='bold')
    ax3.set_xlabel('Nilai Z')
    ax3.set_ylabel('Densitas')
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # ---- Panel 4: Uji T ----
    ax4 = fig.add_subplot(gs[1, 0:2])
    xlim = max(8, abs(t_hit) + 1.5)
    xt = np.linspace(-xlim, xlim, 1000)
    yt = stats.t.pdf(xt, df=df)
    ax4.plot(xt, yt, 'steelblue', lw=2.5, label=f'Distribusi t (df={df})')
    ax4.fill_between(np.linspace(-xlim, -t_crit, 300),
                     stats.t.pdf(np.linspace(-xlim, -t_crit, 300), df=df),
                     alpha=0.55, color='red', label='Tolak H₀')
    ax4.fill_between(np.linspace(t_crit, xlim, 300),
                     stats.t.pdf(np.linspace(t_crit, xlim, 300), df=df),
                     alpha=0.55, color='red')
    ax4.fill_between(np.linspace(-t_crit, t_crit, 500),
                     stats.t.pdf(np.linspace(-t_crit, t_crit, 500), df=df),
                     alpha=0.25, color='green', label='Terima H₀')
    t_plot = max(min(t_hit, xlim - 0.2), -xlim + 0.2)
    ax4.axvline(t_plot, color='purple', lw=2.5, label=f't-hit={t_hit:.4f}')
    ax4.axvline(-t_crit, color='orange', lw=1.8, linestyle='--',
                label=f't-krit=±{t_crit:.4f}')
    ax4.axvline(t_crit, color='orange', lw=1.8, linestyle='--')
    ax4.set_title(f'Uji T – {kep_t}', fontweight='bold')
    ax4.set_xlabel('Nilai t')
    ax4.set_ylabel('Densitas')
    ax4.legend(fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    # ---- Panel 5: Ringkasan Hasil ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary = [
        ["Parameter", "Nilai"],
        ["N", str(n)],
        ["Mean", f"{mean:.4f}"],
        ["Std Dev", f"{s:.4f}"],
        ["CI 95% Bawah", f"{ci_low:.4f}"],
        ["CI 95% Atas", f"{ci_up:.4f}"],
        ["Z-hitung", f"{z_hit:.4f}"],
        ["Z-kritis", f"±{z_crit:.4f}"],
        ["Keputusan Z", kep_z],
        ["t-hitung", f"{t_hit:.4f}"],
        ["t-kritis", f"±{t_crit:.4f}"],
        ["Keputusan T", kep_t],
    ]
    tbl = ax5.table(cellText=summary[1:], colLabels=summary[0],
                    loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2E75B6')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E8F5')
    ax5.set_title('Ringkasan Hasil', fontweight='bold')

    out_path = os.path.join(output_dir, 'statistik_otomatis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGrafik disimpan: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Program Statistik Otomatis dari CSV"
    )
    parser.add_argument('csv_file', nargs='?', default='data_nilai.csv',
                        help='Path file CSV (default: data_nilai.csv)')
    parser.add_argument('--kolom', default='nilai', help='Nama kolom data (default: nilai)')
    parser.add_argument('--mu0', type=float, default=75.0, help='Nilai μ₀ hipotesis (default: 75)')
    parser.add_argument('--sigma', type=float, default=40.0, help='Simpangan baku populasi σ (default: 40)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Tingkat signifikansi (default: 0.05)')
    parser.add_argument('--output', default='.', help='Direktori output grafik (default: .)')
    args = parser.parse_args()

    print("=" * 65)
    print("  PROGRAM STATISTIK OTOMATIS – STATISTIKA DAN PROBABILITAS")
    print("=" * 65)

    # Baca data
    data = baca_csv(args.csv_file, kolom=args.kolom)
    n = len(data)
    print(f"\nFile  : {args.csv_file}")
    print(f"Kolom : {args.kolom}")
    print(f"Data  : {data}")
    print(f"n     : {n}")

    # Confidence Interval
    mean, s, se_ci, t_crit_ci, me, ci_low, ci_up = hitung_ci(data, args.alpha)
    print(f"\n{'─'*60}")
    print("  A. CONFIDENCE INTERVAL (95%)")
    print(f"{'─'*60}")
    print(f"  Mean             = {mean:.4f}")
    print(f"  Standar Deviasi  = {s:.4f}")
    print(f"  Standard Error   = {se_ci:.4f}")
    print(f"  t-kritis (df={n-1}) = {t_crit_ci:.4f}")
    print(f"  Margin of Error  = {me:.4f}")
    print(f"  CI Bawah         = {ci_low:.4f}")
    print(f"  CI Atas          = {ci_up:.4f}")

    # Uji Z
    x_bar, se_z, z_hit, z_crit, p_z, kep_z = uji_z(data, args.mu0, args.sigma, args.alpha)
    print(f"\n{'─'*60}")
    print("  B. UJI Z")
    print(f"{'─'*60}")
    print(f"  H₀: μ = {args.mu0}  |  H₁: μ ≠ {args.mu0}")
    print(f"  σ = {args.sigma}  n = {n}  α = {args.alpha}")
    print(f"  SE               = {se_z:.4f}")
    print(f"  Z-hitung         = {z_hit:.4f}")
    print(f"  Z-kritis         = ±{z_crit:.4f}")
    print(f"  p-value          = {p_z:.6f}")
    print(f"  Keputusan        : {kep_z}")

    # Uji T
    mean_t, s_t, se_t, t_hit, df, t_crit, p_t, kep_t = uji_t(data, args.mu0, args.alpha)
    print(f"\n{'─'*60}")
    print("  C. UJI T")
    print(f"{'─'*60}")
    print(f"  H₀: μ = {args.mu0}  |  H₁: μ ≠ {args.mu0}")
    print(f"  Mean             = {mean_t:.4f}")
    print(f"  Std Dev          = {s_t:.4f}")
    print(f"  SE               = {se_t:.4f}")
    print(f"  t-hitung         = {t_hit:.4f}")
    print(f"  df               = {df}")
    print(f"  t-kritis         = ±{t_crit:.4f}")
    print(f"  p-value          = {p_t:.6f}")
    print(f"  Keputusan        : {kep_t}")
    print("=" * 65)

    # Grafik
    buat_grafik(data, args.mu0, args.sigma, args.alpha, args.output)

if __name__ == "__main__":
    # Jalankan langsung tanpa argumen command line
    import sys
    sys.argv = [
        'statistik_otomatis_csv.py',
        'data_nilai.csv',
        '--mu0', '75',
        '--sigma', '40',
        '--alpha', '0.05',
        '--output', '.'
    ]
    main()