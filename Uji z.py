"""
uji_z.py
Bagian B – Uji Z (Hipotesis Rata-rata Waktu Respons Server)
Mata Kuliah: Statistika dan Probabilitas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ============================================================
# PARAMETER
# ============================================================
mu_0   = 200    # Rata-rata klaim perusahaan (H0)
sigma  = 40     # Simpangan baku populasi (diketahui)
n      = 64     # Ukuran sampel
x_bar  = 210    # Rata-rata sampel
alpha  = 0.05   # Tingkat signifikansi

# ============================================================
# PERHITUNGAN MANUAL
# ============================================================

# Standard Error
se = sigma / (n ** 0.5)                 # 40 / sqrt(64) = 40/8 = 5.0

# Z-hitung
z_hitung = (x_bar - mu_0) / se         # (210-200)/5 = 10/5 = 2.00

# Z-kritis (dua sisi, alpha=0.05 → alpha/2 = 0.025)
z_kritis = stats.norm.ppf(1 - alpha / 2)   # = 1.9600

# p-value (dua sisi)
p_value = 2 * (1 - stats.norm.cdf(abs(z_hitung)))

# Keputusan
if abs(z_hitung) > z_kritis:
    keputusan = "TOLAK H₀"
else:
    keputusan = "GAGAL TOLAK H₀"

# ============================================================
# OUTPUT PERHITUNGAN
# ============================================================
print("=" * 60)
print("  BAGIAN B – UJI Z")
print("=" * 60)
print(f"\nParameter:")
print(f"  μ₀ (klaim) = {mu_0} ms")
print(f"  σ          = {sigma} ms")
print(f"  n          = {n}")
print(f"  x̄          = {x_bar} ms")
print(f"  α          = {alpha}")
print(f"\nHipotesis:")
print(f"  H₀ : μ = {mu_0}  (rata-rata waktu respons = {mu_0} ms)")
print(f"  H₁ : μ ≠ {mu_0}  (rata-rata waktu respons ≠ {mu_0} ms)")
print(f"\nPerhitungan:")
print(f"  SE        = σ/√n = {sigma}/√{n} = {sigma}/{int(n**0.5)} = {se:.4f}")
print(f"  Z-hitung  = (x̄ - μ₀)/SE = ({x_bar} - {mu_0})/{se:.4f} = {z_hitung:.4f}")
print(f"  Z-kritis  = ±{z_kritis:.4f}  (α=0.05, dua sisi)")
print(f"  p-value   = {p_value:.4f}")
print(f"\nKeputusan: {keputusan}")
if keputusan == "TOLAK H₀":
    print(f"  |Z-hitung| = {abs(z_hitung):.4f} > Z-kritis = {z_kritis:.4f}")
    print("  → Data sampel TIDAK mendukung klaim perusahaan hosting.")
print("=" * 60)

# ============================================================
# Visualisasi
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Bagian B – Uji Z Dua Sisi\nWaktu Respons Server Perusahaan Hosting",
             fontsize=14, fontweight='bold')

# Distribusi normal standar
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

ax.plot(x, y, color='steelblue', linewidth=2.5, label='Distribusi Normal Standar')

# Daerah penolakan (kiri)
x_reject_left = np.linspace(-4, -z_kritis, 300)
ax.fill_between(x_reject_left, stats.norm.pdf(x_reject_left),
                alpha=0.6, color='red', label='Daerah Penolakan H₀')

# Daerah penolakan (kanan)
x_reject_right = np.linspace(z_kritis, 4, 300)
ax.fill_between(x_reject_right, stats.norm.pdf(x_reject_right),
                alpha=0.6, color='red')

# Daerah penerimaan
x_accept = np.linspace(-z_kritis, z_kritis, 500)
ax.fill_between(x_accept, stats.norm.pdf(x_accept),
                alpha=0.3, color='green', label='Daerah Penerimaan H₀')

# Garis Z-kritis
ax.axvline(-z_kritis, color='orange', linewidth=2, linestyle='--',
           label=f'Z-kritis = ±{z_kritis:.4f}')
ax.axvline(z_kritis, color='orange', linewidth=2, linestyle='--')

# Posisi Z-hitung
ax.axvline(z_hitung, color='purple', linewidth=3, linestyle='-',
           label=f'Z-hitung = {z_hitung:.4f}')

# Annotations
y_peak = stats.norm.pdf(0)
ax.text(-z_kritis - 0.1, 0.02, f'-{z_kritis:.2f}', ha='right', fontsize=10,
        color='orange', fontweight='bold')
ax.text(z_kritis + 0.1, 0.02, f'+{z_kritis:.2f}', ha='left', fontsize=10,
        color='orange', fontweight='bold')
ax.text(z_hitung + 0.1, y_peak * 0.5, f'Z = {z_hitung:.2f}',
        fontsize=11, color='purple', fontweight='bold')
ax.text(0, y_peak * 0.7, 'Daerah\nPenerimaan\nH₀', ha='center',
        fontsize=10, color='darkgreen', fontweight='bold')
ax.text(-3.2, y_peak * 0.3, 'Tolak\nH₀', ha='center',
        fontsize=10, color='darkred', fontweight='bold')
ax.text(3.2, y_peak * 0.3, 'Tolak\nH₀', ha='center',
        fontsize=10, color='darkred', fontweight='bold')

# Info box
info_text = (f"μ₀ = {mu_0} ms   x̄ = {x_bar} ms\n"
             f"σ = {sigma}   n = {n}   α = {alpha}\n"
             f"Keputusan: {keputusan}")
ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Nilai Z', fontsize=12)
ax.set_ylabel('Densitas Probabilitas', fontsize=12)
ax.set_title('Distribusi Normal – Uji Z Dua Sisi (α = 0.05)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/uji_z.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik disimpan: uji_z.png")