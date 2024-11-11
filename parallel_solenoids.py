# -*- coding: utf-8 -*-

# Eksterne biblioteker
import matplotlib.pyplot as plt
import numpy as np

# Felles definisjoner
from common import bfieldlist, beregn_B_felt, plottingsone

# Konstanter og parametre
I = 1.0  # Strøm gjennom solenoiden
R = 1.0  # Radius av solenoiden
N = 100  # Antall viklinger
L = 2.5  # Lengden på solenoiden
n = 5000  # Antall punkter
y0 = 2.0  # Avstand fra xz-planet

# Størrelsene på planene
steg = 100  # Antall steg
line_h = 1.5 * (y0 + 2 * R)  # Horisontal størrelse på plottet
line_v = 1.5 * L  # Verdikal størrelse på plottet

# - koordinatene på planene
px = np.linspace(-line_h, line_h, steg)
py = np.linspace(-line_h, line_h, steg)
pz = np.linspace(-line_v, line_v, steg)

# - xy-planet
_x, _y = np.meshgrid(px, py)
_z = np.zeros_like(_x)
p_xy = [_x, _y, _z]

# - yz-planet
_y, _z = np.meshgrid(py, pz)
_x = np.zeros_like(_y)
p_yz = [_x, _y, _z]

# - xz-planet
_x, _z = np.meshgrid(px, pz)
_y = np.zeros_like(_z)
p_xz = [_x, _y, _z]

del _x, _y, _z

# Vinkel for en enkel solenoide
theta = np.linspace(0, 2 * np.pi * N, n)

# Koordinatene til solenoide nr. 1
# - rotasjon mot klokka
x1 = R * np.cos(theta)
y1 = (y0 + R) + R * np.sin(theta)
z1 = L * (theta - N * np.pi) / (2 * np.pi * N)

k1 = np.column_stack((x1, y1, z1))
I1 = I

# Koordinatene til solenoide nr 2.
# - speilvendt om xy-planet,
#   rotasjon med klokka
x2 = R * np.cos(theta)
y2 = -(y0 + R) + R * np.sin(theta)
z2 = z1

k2 = np.column_stack((x2, y2, z2))
I2 = I

# Liste med planer
planer = [
    (p_xz, [0, 2], ["XZ", "x", "z"], "b"),
    (p_yz, [1, 2], ["YZ", "y", "z"], "g"),
    (p_xy, [0, 1], ["XY", "x", "y"], "r"),
]

# Regn ut feltstyrken fra begge solenoider, sett
# fra xz-, yz- og xy-planet
for plan, indeks, navn, farge in planer:
    # Regn ut feltstyrken fra hver solenoide
    B1 = beregn_B_felt(plan[0], plan[1], plan[2], k1, I1)
    B2 = beregn_B_felt(plan[0], plan[1], plan[2], k2, I2)

    # Hent fram relevante komponenter og regn ut
    # netto feltstyrke
    B1_ax1, B1_ax2 = B1[indeks[0]], B1[indeks[1]]
    B2_ax1, B2_ax2 = B2[indeks[0]], B2[indeks[1]]
    B_ax1, B_ax2 = B1_ax1 + B2_ax1, B1_ax2 + B2_ax2
    ax1, ax2 = plan[indeks[0]], plan[indeks[1]]

    # Regn ut feltstyrken i origo og skriv den ut
    origo = np.zeros(3)
    B_origo = bfieldlist(origo, k1, I1) + bfieldlist(origo, k2, I2)

    print("|B(0, 0, 0)| = %.3f uT" % (np.linalg.norm(B_origo) * 1.0e6))

    # Plott feltstyrken ved bruk av metoden `plottingsone`
    plottingsone(B_ax1, B_ax2, ax1, ax2, navn, farge, tol=1.0e-7)

plt.show()
