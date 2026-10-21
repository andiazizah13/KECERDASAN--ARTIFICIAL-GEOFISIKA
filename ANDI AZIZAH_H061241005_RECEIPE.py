
"""
SIMULASI DIFUSI PANAS BAWAH PERMUKAAN
Recipe: R-MOD-001 | Case Study 23
Metode: Finite Difference Method (Explicit FTCS)

Catatan:
Data merupakan simulasi sintetis, bukan data pengukuran lapangan.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PARAMETER SIMULASI
# ============================================================

KAPPA = 1.0e-6       # Difusivitas termal (m^2/s)
DEPTH = 5.0          # Kedalaman model (m)
DZ = 0.1             # Jarak antar-node (m)
DURATION_DAYS = 30   # Durasi simulasi (hari)
DT = 3600.0          # Langkah waktu (detik = 1 jam)

T_INITIAL = 10.0     # Temperatur awal (derajat Celsius)
T_SURFACE = 25.0     # Temperatur permukaan (derajat Celsius)
T_BOTTOM = 10.0      # Temperatur batas bawah (derajat Celsius)

OUTPUT_DIR = "output_heat_diffusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. MEMBENTUK GRID KEDALAMAN DAN WAKTU
# ============================================================

z = np.linspace(0.0, DEPTH, int(round(DEPTH / DZ)) + 1)
nz = len(z)

duration_seconds = DURATION_DAYS * 24 * 3600
nt = int(round(duration_seconds / DT))

time_days = np.arange(nt + 1) * DT / 86400.0

# Rasio stabilitas FTCS
r = KAPPA * DT / DZ**2

print("=" * 55)
print("SIMULASI DIFUSI PANAS BAWAH PERMUKAAN")
print("=" * 55)
print(f"Jumlah node kedalaman : {nz}")
print(f"Jumlah langkah waktu  : {nt}")
print(f"Rasio stabilitas (r)  : {r:.3f}")
print("Syarat stabilitas     : r <= 0.5")

if r > 0.5:
    raise ValueError(
        "Simulasi tidak stabil. Kurangi DT atau perbesar DZ."
    )

# ============================================================
# 3. KONDISI AWAL DAN BATAS
# ============================================================

T = np.full(nz, T_INITIAL, dtype=float)

# Batas temperatur tetap
T[0] = T_SURFACE
T[-1] = T_BOTTOM

# Menyimpan evolusi temperatur
temperature = np.zeros((nt + 1, nz), dtype=float)
temperature[0, :] = T.copy()

# ============================================================
# 4. SIMULASI DIFUSI PANAS: METODE FTCS
# ============================================================

for n in range(nt):
    T_new = T.copy()

    # Persamaan beda hingga eksplisit 1D
    T_new[1:-1] = (
        T[1:-1]
        + r * (T[2:] - 2.0 * T[1:-1] + T[:-2])
    )

    # Mempertahankan kondisi batas
    T_new[0] = T_SURFACE
    T_new[-1] = T_BOTTOM

    T = T_new
    temperature[n + 1, :] = T

print("\nSimulasi berhasil dijalankan.")
print(f"Temperatur permukaan : {T[0]:.2f} °C")
print(f"Temperatur batas bawah: {T[-1]:.2f} °C")

# ============================================================
# 5. MENYIMPAN PROFIL TEMPERATUR KE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR, "profil_temperatur_hari30.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Kedalaman_m", "Temperatur_C"])

    for depth, temp in zip(z, temperature[-1, :]):
        writer.writerow([f"{depth:.2f}", f"{temp:.6f}"])

print(f"Data CSV disimpan: {csv_path}")

# ============================================================
# 6. GRAFIK 1: HEATMAP TEMPERATUR
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

mesh = ax.pcolormesh(
    time_days,
    z,
    temperature.T,
    shading="auto",
    cmap="inferno"
)

ax.set_title(
    "Simulasi Difusi Panas Bawah Permukaan\n"
    "Data Sintetis - Bukan Data Lapangan"
)
ax.set_xlabel("Waktu (hari)")
ax.set_ylabel("Kedalaman (m)")
ax.invert_yaxis()

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Temperatur (°C)")

ax.grid(alpha=0.2)
fig.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR, "heatmap_difusi_panas.png"
)
fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
print(f"Heatmap disimpan: {heatmap_path}")

plt.show()
plt.close(fig)

# ============================================================
# 7. GRAFIK 2: PROFIL TEMPERATUR HARI KE-30
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    temperature[-1, :],
    z,
    linewidth=2.0,
    label=f"Hari ke-{DURATION_DAYS}"
)

ax.set_title(
    "Profil Temperatur terhadap Kedalaman\n"
    "Data Sintetis"
)
ax.set_xlabel("Temperatur (°C)")
ax.set_ylabel("Kedalaman (m)")
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

profile_path = os.path.join(
    OUTPUT_DIR, "profil_temperatur_hari30.png"
)
fig.savefig(profile_path, dpi=200, bbox_inches="tight")
print(f"Grafik profil disimpan: {profile_path}")

plt.show()
plt.close(fig)

# ============================================================
# 8. RINGKASAN DAN VALIDASI
# ============================================================

checks = {
    "Stabilitas numerik": r <= 0.5,
    "Semua temperatur bernilai finite":
        bool(np.isfinite(temperature).all()),
    "Batas permukaan tetap":
        bool(np.allclose(temperature[:, 0], T_SURFACE)),
    "Batas bawah tetap":
        bool(np.allclose(temperature[:, -1], T_BOTTOM)),
    "Temperatur berada dalam rentang batas":
        bool(
            temperature.min() >= min(
                T_INITIAL, T_SURFACE, T_BOTTOM
            ) - 1e-9
            and temperature.max() <= max(
                T_INITIAL, T_SURFACE, T_BOTTOM
            ) + 1e-9
        )
}

print("\n" + "=" * 55)
print("HASIL VALIDASI")
print("=" * 55)

for name, passed in checks.items():
    status = "LULUS" if passed else "GAGAL"
    print(f"{name}: {status}")

# ============================================================
# 9. TEMPERATUR PADA BEBERAPA KEDALAMAN
# ============================================================

print("\nTemperatur pada beberapa kedalaman, hari ke-30:")

for target_depth in [0, 1, 2, 3, 4, 5]:
    idx = int(round(target_depth / DZ))
    value = temperature[-1, idx]

    print(
        f"Kedalaman {z[idx]:.1f} m: "
        f"{value:.3f} °C"
    )

print("\nSemua output tersimpan di folder:", OUTPUT_DIR)
print("Simulasi selesai.")
