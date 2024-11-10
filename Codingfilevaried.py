import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange  # For kovertering til C++ for enklere kjøring

# Konstanter
mu0 = 4 * np.pi * 1e-7  # Permeabiliteten i vakuum
I = 1.0  # Strøm gjennom solenoiden
R = 1.0  # Radius av solenoiden
N = 100  # Antall viklinger
L = 5.0  # Lengden på solenoiden
n = N / L  # Antall viklinger per lengdeenhet

# Parametere for solenoiden
antall_viklinger = N
radius = R
lengde = L
antall_punkter = 5000

# Oppretter koordinatene til solenoiden
theta = np.linspace(0, 2 * np.pi * antall_viklinger, antall_punkter)
z = np.linspace(-lengde / 2, lengde / 2, antall_punkter)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
koordinater = np.column_stack((x, y, z))

# Numba-optimalisert bfieldlist-funksjon
@njit
def bfieldlist(r, koordinater):
    # Bruker den B-felt fra pensum Chapter 11.1
    # "Elementary Electromagnetism Using Python"
    B = np.zeros(3)
    N = koordinater.shape[0]
    for i in range(N):
        i0 = i
        i1 = (i + 1) % N  # Sikrer at vi går tilbake til start
        midtpunkt = 0.5 * (
            koordinater[i1] + koordinater[i0]
        )  # Midtpunktet av segmentet
        R_vec = r - midtpunkt
        dlv = koordinater[i1] - koordinater[i0]  # Differensial lengdevektor
        norm_R = np.linalg.norm(R_vec)
        dB = (mu0 * I / (4 * np.pi)) * np.cross(dlv, R_vec) / norm_R ** 3
        B += dB
    return B


# Numba-optimalisert funksjon for å beregne magnetfeltet over gridet
# Denne funskjonen gjør at det er mulig å paraelle regne B felt raskere
@njit(parallel=True)
def beregn_B_felt(X, Y, Z, koordi):
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.array([X[i, j], Y[i, j], Z[i, j]])
            B = bfieldlist(r, koordi)
            Bx[i, j] = B[0]
            By[i, j] = B[1]
            Bz[i, j] = B[2]
    return Bx, By, Bz


# Funksjon for plotting
def plottingsone(B1, B2, axis1, axis2, navn, farge):
    plt.figure(figsize=(8, 6))
    plt.streamplot(axis1, axis2, B1, B2, color=farge, density=1.5)

    # Beregn størrelsen på magnetfeltet
    B_magnitude = np.sqrt(B1 ** 2 + B2 ** 2)
    errorsone = 1e-9  # Terskelverdi for når feltet er tilnærmet 0
    zero_field = B_magnitude <= errorsone  # Områder med felt under terskelverdien

    # Plotter områder med null felt
    plt.contourf(axis1, axis2, zero_field, levels=[1e-9, 1], colors="purple", alpha=0.5)

    # Legger til fargebar
    cbar = plt.colorbar()
    cbar.set_label("0=B-felt")

    plt.xlabel(f"{navn[1]} (m)")
    plt.ylabel(f"{navn[2]} (m)")
    plt.title(f"B-felt rundt en solenoide i {navn[0]}-planet")
    plt.grid(True)


# Lager standard på alle planene
steg = 50
line = 5
x = np.linspace(-line, line, steg)
z = np.linspace(-line, line, steg)
y = np.linspace(-line, line, steg)

# --- for XZ-planet--- med blå
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)

# Beregner magnetfeltet ved hvert punkt i gridet
Bx, By, Bz = beregn_B_felt(X, Y, Z, koordinater)
navnXZ = ["XZ", "x", "z"]
plottingsone(Bx, Bz, X, Z, navnXZ, "b")
# Tegner solenoiden
plt.fill_between([-R, R], -L / 2, L / 2, color="gray", alpha=0.3)

# --- for XY-planet--- med rød
# Definerer grid i XY-planet
X_, Y_ = np.meshgrid(x, y)
Z = np.zeros_like(X_)

# Beregner magnetfeltet ved hvert punkt i gridet
Bx, By, Bz = beregn_B_felt(X_, Y_, Z, koordinater)
navnXY = ["XY", "x", "y"]
plottingsone(Bx, By, X_, Y_, navnXY, "r")
# Tegner solenoiden
sirkel = plt.Circle((0, 0), R, color="gray", alpha=0.3)
plt.gca().add_artist(sirkel)
plt.show()
