import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Konstanter
mu0 = 4 * np.pi * 1e-7  # Permeabiliteten i vakuum
I = 1.0  # Strøm gjennom solenoiden
N = 100  # Antall viklinger
# Parametere for solenoiden
antall_punkter = 1000
# Numba-optimalisert bfieldlist-funksjon
@njit
def bfieldlist(r, koordinater):
    B = np.zeros(3)
    Np = koordinater.shape[0]
    for i in range(Np):
        i0 = i
        i1 = (i + 1) % Np  # Sikrer at vi går tilbake til start
        midtpunkt = 0.5 * (koordinater[i1] + koordinater[i0])  # Midtpunktet av segmentet
        R_vec = r - midtpunkt
        dlv = koordinater[i1] - koordinater[i0]  # Differensial lengdevektor
        norm_R = np.sqrt(R_vec[0]**2 + R_vec[1]**2 + R_vec[2]**2)
        if norm_R == 0:
            continue
        kryss = np.array([
            dlv[1]*R_vec[2] - dlv[2]*R_vec[1],
            dlv[2]*R_vec[0] - dlv[0]*R_vec[2],
            dlv[0]*R_vec[1] - dlv[1]*R_vec[0]
        ])
        dB = (mu0 * I / (4 * np.pi)) * kryss / norm_R**3
        B += dB
    return B

# Numba-optimalisert funksjon for å beregne magnetfeltet over gridet
@njit(parallel=True)
def beregn_B_felt(X, Y, Z, koordinater):
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    for i in prange(X.shape[0]):
        for j in range(X.shape[1]):
            r = np.array([X[i, j], Y[i, j], Z[i, j]])
            B = bfieldlist(r, koordinater)
            Bx[i, j] = B[0]
            By[i, j] = B[1]
            Bz[i, j] = B[2]
    return Bx, By, Bz

# Funksjon for plotting
def plottingsone(B1, B2, axis1, axis2, navn, farge, R, L):
    plt.figure(figsize=(8, 6))
    plt.streamplot(axis1, axis2, B1, B2, color=farge, density=1.5)
    
    # Beregn størrelsen på magnetfeltet
    B_magnitude = np.sqrt(B1**2 + B2**2)
    errorsone = 1e-9  # Terskelverdi for når feltet er tilnærmet 0
    zero_field = B_magnitude <= errorsone  # Områder med felt under terskelverdien

    # Plotter områder med null felt
    plt.contourf(axis1, axis2, zero_field, levels=[1e-9, 1], colors='purple', alpha=0.5)

    # Legger til fargebar
    cbar = plt.colorbar()
    cbar.set_label('0=B-felt')

    plt.xlabel(f'{navn[1]} (m)')
    plt.ylabel(f'{navn[2]} (m)')
    plt.title(f'B-felt rundt en solenoide i {navn[0]}-planet (R={R}, L={L})')
    plt.grid(True)

# Definerer grid
steg = 50
line = 5
x = np.linspace(-line, line, steg)
z = np.linspace(-line, line, steg)
y = np.linspace(-line, line, steg)

# *** START på for-løkke over forskjellige solenoidstørrelser ***
# Liste over forskjellige radius og lengder
radius_liste = [0.5, 1.0, 1.5]
lengde_liste = [3.0, 5.0, 7.0]

for R in radius_liste:
    for L in lengde_liste:
        # Oppdaterer antall viklinger per lengdeenhet
        n = N / L
        radius = R
        lengde = L
        antall_viklinger = N
        # Oppretter koordinatene til solenoiden
        theta = np.linspace(0, 2 * np.pi * antall_viklinger, antall_punkter)
        z_solenoid = np.linspace(-lengde / 2, lengde / 2, antall_punkter)
        x_solenoid = radius * np.cos(theta)
        y_solenoid = radius * np.sin(theta)
        koordinater = np.column_stack((x_solenoid, y_solenoid, z_solenoid))

        # *** Plot i XZ-planet ***
        # Definerer grid i XZ-planet
        X_grid, Z_grid = np.meshgrid(x, z)
        Y_grid = np.zeros_like(X_grid)

        # Beregner magnetfeltet ved hvert punkt i gridet
        Bx, By, Bz = beregn_B_felt(X_grid, Y_grid, Z_grid, koordinater)

        navnXZ = ["XZ", "x", "z"]
        plottingsone(Bx, Bz, X_grid, Z_grid, navnXZ, "b", R, L)
        # Tegner solenoiden
        plt.fill_between([-R, R], -L/2, L/2, color='gray', alpha=0.3)


        # *** Plot i XY-planet ***
        # Definerer grid i XY-planet
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = np.zeros_like(X_grid)

        # Beregner magnetfeltet ved hvert punkt i gridet
        Bx, By, Bz = beregn_B_felt(X_grid, Y_grid, Z_grid, koordinater)

        navnXY = ["XY", "x", "y"]
        plottingsone(Bx, By, X_grid, Y_grid, navnXY, "r", R, L)
        # Tegner solenoiden
        sirkel = plt.Circle((0, 0), R, color='gray', alpha=0.3)
        plt.gca().add_artist(sirkel)
        plt.axis('equal')

plt.show()
# *** SLUTT på for-løkke over forskjellige solenoidstørrelser ***
