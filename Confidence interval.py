"""
confidence_interval.py
Bagian A – Confidence Interval
Mata Kuliah: Statistika dan Probabilitas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ============================================================
# DATA
# ============================================================
data = [78, 82, 75, 80, 79,
        81, 77, 83, 76, 84,
        79, 80, 78, 82, 77,
        81, 79, 80, 78, 83]

n = len(data)
alpha = 0.05
confidence = 0.95

# ============================================================
# PERHITUNGAN MANUAL
# ============================================================

# Mean
mean = sum(data) / n  # = 79.6

# Standar Deviasi Sampel
variance = sum((xi - mean) ** 2 for xi in data) / (n - 1)
std_dev = variance ** 0.5

# Standard Error
se = std_dev / (n ** 0.5)

# t-kritis (df = n-1 = 19)
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

# Margin of Error
me = t_critical * se

# Confidence Interval
ci_lower = mean - me
ci_upper = mean + me

# ============================================================
# OUTPUT PERHITUNGAN
# ============================================================
print("=" * 60)
print("  BAGIAN A – CONFIDENCE INTERVAL (95%)")
print("=" * 60)
print(f"\nData: {data}")
print(f"\nJumlah Data (n)           : {n}")
print(f"Sum Data                  : {sum(data)}")
print(f"Mean (x̄)                  : {mean:.4f}")
print(f"Variansi Sampel (s²)      : {variance:.4f}")
print(f"Standar Deviasi (s)       : {std_dev:.4f}")
print(f"Standard Error (SE)       : {se:.4f}")
print(f"t-kritis (df=19, α/2=0.025): {t_critical:.4f}")
print(f"Margin of Error (ME)      : {me:.4f}")
print(f"\nInterval Kepercayaan 95%:")
print(f"  Batas Bawah (Lower)     : {ci_lower:.4f}")
print(f"  Batas Atas  (Upper)     : {ci_upper:.4f}")
print("=" * 60)

# ============================================================
# VISUALISASI
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Bagian A – Confidence Interval 95%\nNilai Kuis Mahasiswa Statistika dan Probabilitas",
             fontsize=14, fontweight='bold')

# --- Plot 1: Distribusi Data ---
ax1 = axes[0]
ax1.hist(data, bins=8, color='steelblue', edgecolor='white', alpha=0.85, linewidth=1.2)
ax1.axvline(mean, color='red', linewidth=2.5, linestyle='-', label=f'Mean = {mean:.4f}')
ax1.axvline(ci_lower, color='orange', linewidth=2, linestyle='--', label=f'CI Bawah = {ci_lower:.4f}')
ax1.axvline(ci_upper, color='green', linewidth=2, linestyle='--', label=f'CI Atas = {ci_upper:.4f}')
ax1.axvspan(ci_lower, ci_upper, alpha=0.15, color='yellow', label='Interval CI 95%')
ax1.set_xlabel('Nilai Mahasiswa', fontsize=12)
ax1.set_ylabel('Frekuensi', fontsize=12)
ax1.set_title('Histogram Distribusi Nilai', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Plot 2: Confidence Interval ---
ax2 = axes[1]
# Distribusi t
df = n - 1
x = np.linspace(mean - 4 * se, mean + 4 * se, 500)
y = stats.norm.pdf(x, loc=mean, scale=se)

ax2.plot(x, y, 'steelblue', linewidth=2.5, label='Distribusi Sampling')

# Shade CI area
x_fill = np.linspace(ci_lower, ci_upper, 300)
y_fill = stats.norm.pdf(x_fill, loc=mean, scale=se)
ax2.fill_between(x_fill, y_fill, alpha=0.35, color='orange', label=f'CI 95% [{ci_lower:.4f}, {ci_upper:.4f}]')

# Mean line
ax2.axvline(mean, color='red', linewidth=2.5, linestyle='-', label=f'Mean = {mean:.4f}')
ax2.axvline(ci_lower, color='orange', linewidth=2, linestyle='--')
ax2.axvline(ci_upper, color='green', linewidth=2, linestyle='--')

# Annotations
ax2.annotate(f'x̄ = {mean:.4f}', xy=(mean, max(y) * 0.5),
             xytext=(mean + 0.6, max(y) * 0.6),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')
ax2.text(ci_lower - 0.05, max(y) * 0.1, f'{ci_lower:.4f}', ha='right', fontsize=9, color='orange')
ax2.text(ci_upper + 0.05, max(y) * 0.1, f'{ci_upper:.4f}', ha='left', fontsize=9, color='green')

ax2.set_xlabel('Nilai Rata-rata', fontsize=12)
ax2.set_ylabel('Densitas Probabilitas', fontsize=12)
ax2.set_title('Distribusi Sampling & Interval Kepercayaan 95%', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/confidence_interval.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik disimpan: confidence_interval.png")