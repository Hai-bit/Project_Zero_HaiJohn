import numpy as np
import matplotlib.pyplot as plt

from common import beregn_B_felt
# Konstanter
mu0 = 4 * np.pi * 1e-7  # Permeabiliteten i vakuum
I = 1.0  # Strøm gjennom solenoiden
N = 100  # Antall viklinger
# Parametere for solenoiden for med sikker nummer
antall_punkter = 5000


# Funksjon for plotting
def plottingsone(B1, B2, axis1, axis2, navn, farge, R, L):
    plt.figure(figsize=(8, 6))

    B_magnitude = np.sqrt(B1 ** 2 + B2 ** 2)
    contour = plt.contourf(axis1, axis2, B_magnitude, levels=50, cmap="viridis")
    cbar = plt.colorbar(contour, ax=plt.gca())
    cbar.set_label("Magnetfeltstyrke (T)")
    plt.streamplot(axis1, axis2, B1, B2, color=farge, density=1.5)

    # Beregn størrelsen på magnetfeltet
    errorsone = 1e-9  # Terskelverdi for når feltet er tilnærmet 0
    zero_field = B_magnitude <= errorsone  # Områder med felt under terskelverdien
    # Plotter områder med null felt , endre med grønn siden den ikke er ikke bruk
    plt.contourf(
        axis1, axis2, zero_field, levels=[1e-9, 0.1], colors="green", alpha=0.5
    )

    # Legger til fargebar
    cbar = plt.colorbar()
    cbar.set_label("0=B-felt")

    plt.xlabel(f"{navn[1]} (m)")
    plt.ylabel(f"{navn[2]} (m)")
    plt.title(f"B-felt rundt en solenoide i {navn[0]}-planet (R={R}, L={L})")
    plt.grid(True)


# Definerer grid
steg = 50
line = 5
x = np.linspace(-line, line, steg)
z = np.linspace(-line, line, steg)
y = np.linspace(-line, line, steg)


""" 
Endringer på lengde eller radiusen gjøres på 
radius_liste og lengde_liste
"""
# Liste over forskjellige radius og lengder samt standarer
radius_liste = [1.0, 2.0, 3.0] # endre disse for endre r
standard_radius = [1.0]
lengde_liste = [3.0, 5.0, 7.0] # endre disse for endre lengden
standard_lengde = [5.0]

def cal(L_liste,
    R_liste): 
    """
     Hoved kalklulasjonene til å plotte ut B-feltet
    R_liste: array
        Listen til vedlagt radius
    L_liste : array
        Listen til vedlagt lengde
    """
    for L in L_liste:
        for R in R_liste:
            # Oppdaterer antall viklinger per lengdeenhet
            n = N / L
            radius = R
            lengde = L
            # Oppretter koordinatene til solenoiden
            theta = np.linspace(0, 2 * np.pi * N, antall_punkter)
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
            plt.fill_between([-R, R], -L / 2, L / 2, color="gray", alpha=0.3)

            # *** Plot i XY-planet ***
            # Definerer grid i XY-planet
            X_grid, Y_grid = np.meshgrid(x, y)
            Z_grid = np.zeros_like(X_grid)

            # Beregner magnetfeltet ved hvert punkt i gridet
            Bx, By, Bz = beregn_B_felt(X_grid, Y_grid, Z_grid, koordinater)

            navnXY = ["XY", "x", "y"]
            plottingsone(Bx, By, X_grid, Y_grid, navnXY, "r", R, L)
            # Tegner solenoiden
            sirkel = plt.Circle((0, 0), R, color="gray", alpha=0.3)
            plt.gca().add_artist(sirkel)
            plt.axis("equal")
"""
Kalkulerer B feltet ved forskjellige typer
"""
cal(lengde_liste,standard_radius)
cal(standard_lengde,radius_liste)
plt.show()
