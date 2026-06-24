"""
uji_t.py
Bagian C – Uji T (Metode Pembelajaran Baru)
Mata Kuliah: Statistika dan Probabilitas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# DATA
# ============================================================
data = [78, 80, 75, 82,
        79, 81, 77, 83,
        76, 80, 79, 82,
        78, 81, 77, 84]

mu_0  = 75      # Rata-rata standar sebelumnya
alpha = 0.05    # Tingkat signifikansi
n     = len(data)

# ============================================================
# PERHITUNGAN MANUAL
# ============================================================

# Mean
mean = sum(data) / n    # = 79.5

# Standar deviasi sampel
variance = sum((xi - mean) ** 2 for xi in data) / (n - 1)
std_dev  = variance ** 0.5

# Standard Error
se = std_dev / (n ** 0.5)

# t-hitung
t_hitung = (mean - mu_0) / se

# Derajat kebebasan
df = n - 1

# t-kritis (dua sisi)
t_kritis = stats.t.ppf(1 - alpha / 2, df=df)

# p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_hitung), df=df))

# Keputusan
if abs(t_hitung) > t_kritis:
    keputusan = "TOLAK H₀"
else:
    keputusan = "GAGAL TOLAK H₀"

# ============================================================
# OUTPUT PERHITUNGAN
# ============================================================
print("=" * 60)
print("  BAGIAN C – UJI T")
print("=" * 60)
print(f"\nData: {data}")
print(f"\nParameter:")
print(f"  μ₀ (standar) = {mu_0}")
print(f"  n             = {n}")
print(f"  α             = {alpha}")
print(f"\nPerhitungan:")
print(f"  Sum data  = {sum(data)}")
print(f"  Mean (x̄) = {mean:.4f}")
print(f"  Variansi  = {variance:.4f}")
print(f"  Std Dev   = {std_dev:.4f}")
print(f"  SE        = s/√n = {std_dev:.4f}/√{n} = {std_dev:.4f}/{n**0.5:.4f} = {se:.4f}")
print(f"\nHipotesis:")
print(f"  H₀ : μ = {mu_0}  (rata-rata tidak berbeda dari standar)")
print(f"  H₁ : μ ≠ {mu_0}  (rata-rata berbeda dari standar)")
print(f"\nUji Statistik:")
print(f"  t-hitung = (x̄ - μ₀)/SE = ({mean:.4f} - {mu_0})/{se:.4f} = {t_hitung:.4f}")
print(f"  df       = n-1 = {n}-1 = {df}")
print(f"  t-kritis = ±{t_kritis:.4f}  (α=0.05, df={df}, dua sisi)")
print(f"  p-value  = {p_value:.6f}")
print(f"\nKeputusan: {keputusan}")
if keputusan == "TOLAK H₀":
    print(f"  |t-hitung| = {abs(t_hitung):.4f} > t-kritis = {t_kritis:.4f}")
    print("  → Rata-rata nilai mahasiswa BERBEDA secara signifikan dari standar μ₀=75.")
print("=" * 60)

# ============================================================
# VISUALISASI
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Bagian C – Uji T Dua Sisi\nEvaluasi Metode Pembelajaran Baru (μ₀ = 75)",
             fontsize=14, fontweight='bold')

# Distribusi t
x = np.linspace(-8, 8, 1000)
y = stats.t.pdf(x, df=df)

ax.plot(x, y, color='steelblue', linewidth=2.5, label=f'Distribusi t (df={df})')

# Daerah penolakan kiri
x_reject_left = np.linspace(-8, -t_kritis, 300)
ax.fill_between(x_reject_left, stats.t.pdf(x_reject_left, df=df),
                alpha=0.6, color='red', label='Daerah Penolakan H₀')

# Daerah penolakan kanan
x_reject_right = np.linspace(t_kritis, 8, 300)
ax.fill_between(x_reject_right, stats.t.pdf(x_reject_right, df=df),
                alpha=0.6, color='red')

# Daerah penerimaan
x_accept = np.linspace(-t_kritis, t_kritis, 500)
ax.fill_between(x_accept, stats.t.pdf(x_accept, df=df),
                alpha=0.3, color='green', label='Daerah Penerimaan H₀')

# Garis t-kritis
ax.axvline(-t_kritis, color='orange', linewidth=2, linestyle='--',
           label=f't-kritis = ±{t_kritis:.4f}')
ax.axvline(t_kritis, color='orange', linewidth=2, linestyle='--')

# Posisi t-hitung
ax.axvline(t_hitung, color='purple', linewidth=3, linestyle='-',
           label=f't-hitung = {t_hitung:.4f}')

# Annotations
y_peak = stats.t.pdf(0, df=df)
ax.text(-t_kritis - 0.15, 0.01, f'-{t_kritis:.3f}', ha='right',
        fontsize=10, color='orange', fontweight='bold')
ax.text(t_kritis + 0.15, 0.01, f'+{t_kritis:.3f}', ha='left',
        fontsize=10, color='orange', fontweight='bold')
ax.text(t_hitung + 0.25, y_peak * 0.25, f't = {t_hitung:.4f}\n(di luar frame kiri)',
        fontsize=9, color='purple', fontweight='bold')
ax.text(0, y_peak * 0.65, 'Daerah\nPenerimaan H₀', ha='center',
        fontsize=10, color='darkgreen', fontweight='bold')
ax.text(-5.5, y_peak * 0.2, 'Tolak H₀', ha='center',
        fontsize=10, color='darkred', fontweight='bold')
ax.text(5.5, y_peak * 0.2, 'Tolak H₀', ha='center',
        fontsize=10, color='darkred', fontweight='bold')

# Info box
info_text = (f"μ₀ = {mu_0}   x̄ = {mean:.4f}\n"
             f"s = {std_dev:.4f}   n = {n}   df = {df}\n"
             f"α = {alpha}   p-value = {p_value:.6f}\n"
             f"Keputusan: {keputusan}")
ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Nilai t', fontsize=12)
ax.set_ylabel('Densitas Probabilitas', fontsize=12)
ax.set_title(f'Distribusi t (df={df}) – Uji T Dua Sisi (α = 0.05)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-8, 8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/uji_t.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik disimpan: uji_t.png")