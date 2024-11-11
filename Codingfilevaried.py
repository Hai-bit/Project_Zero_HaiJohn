# -*- coding: utf-8 -*-

# Eksterne biblioteker
import numpy as np
import matplotlib.pyplot as plt

# Felles definisjoner
from common import beregn_B_felt, plottingsone

# Konstanter
I = 1.0  # Strøm gjennom solenoiden
N = 100  # Antall viklinger
# Parametere for solenoiden for med sikker nummer
antall_punkter = 5000

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
radius_liste = [1.0, 2.0, 3.0]  # endre disse for endre r
standard_radius = [1.0]
lengde_liste = [3.0, 5.0, 7.0]  # endre disse for endre lengden
standard_lengde = [5.0]


def cal(L_liste, R_liste):
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
            Bx, By, Bz = beregn_B_felt(X_grid, Y_grid, Z_grid, koordinater, I)

            navnXZ = ["XZ", "x", "z"]
            plottingsone(Bx, Bz, X_grid, Z_grid, navnXZ, "b", radius, lengde)
            # Tegner solenoiden
            plt.fill_between([-R, R], -L / 2, L / 2, color="gray", alpha=0.3)

            # *** Plot i XY-planet ***
            # Definerer grid i XY-planet
            X_grid, Y_grid = np.meshgrid(x, y)
            Z_grid = np.zeros_like(X_grid)

            # Beregner magnetfeltet ved hvert punkt i gridet
            Bx, By, Bz = beregn_B_felt(X_grid, Y_grid, Z_grid, koordinater, I)

            navnXY = ["XY", "x", "y"]
            plottingsone(Bx, By, X_grid, Y_grid, navnXY, "r", R, L)
            # Tegner solenoiden
            sirkel = plt.Circle((0, 0), R, color="gray", alpha=0.3)
            plt.gca().add_artist(sirkel)
            plt.axis("equal")


"""
Kalkulerer B feltet ved forskjellige typer
"""
cal(lengde_liste, standard_radius)
cal(standard_lengde, radius_liste)
plt.show()
