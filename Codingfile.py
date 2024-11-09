#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Grunnleggende biblioteker
import numpy as np
import matplotlib.pyplot as plt

# Felles definisjoner
from common import beregn_B_felt, plottingsone

# Konstanter
I = 1.0  # Strøm gjennom solenoiden
R = 1.0  # Radius av solenoiden
N = 100  # Antall viklinger
L = 5.0  # Lengden på solenoiden
n = N / L  # Antall viklinger per lengdeenhet

# Parametere for solenoiden
antall_viklinger = N
radius = R
lengde = L
antall_punkter = 10000

# Oppretter koordinatene til solenoiden
theta = np.linspace(0, 2 * np.pi * antall_viklinger, antall_punkter)
z = np.linspace(-lengde / 2, lengde / 2, antall_punkter)
x = radius * np.cos(theta)
y = radius * np.sin(theta)
koordinater = np.column_stack((x, y, z))

# Lager standard på alle planene
steg = 50
line = 4
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

# --- for YZ-planet--- med grønn
# Definerer grid i YZ-planet
Y, Z_ = np.meshgrid(y, z)
X = np.zeros_like(Y)

# Beregner magnetfeltet ved hvert punkt i gridet
Bx, By, Bz = beregn_B_felt(X, Y, Z_, koordinater)
navnYZ = ["YZ", "y", "z"]
plottingsone(By, Bz, Y, Z_, navnYZ, "g")
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
