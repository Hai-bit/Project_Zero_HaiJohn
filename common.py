#!/usr/bin/env python3
# -*- code: utf-8 -*-

"""
En bibliotek som inneholder felles rutiner
og definisjoner som tas i bruk i numeriske
utregninger for Computational Essay i
FYS1120 - semester H 2024
"""


# Eksterne biblioteker
import numpy as np                  # Vektorisert regning
import matplotlib.pyplot as plt     # Plotting
from numba import njit, prange      # JIT-kompilering (raskere kjøring)


# Konstanter
mu0 = 4 * np.pi * 1.0e-7            # Permeabilitiet i vakuum (T * m / A)
I = 1.0                             # Strømmen gjennom solenoiden (A)

# --- Metoder --- #

# Metode for å regne ut magnetisk feltstyrke
# fra en strømførende leder
# - bruker eksemplet fra pensum: Chapter 11.1
# "Elementary Electromagnetism Using Python"
@njit
def bfieldlist(r: np.ndarray, koordinater: np.ndarray, i: float = I) -> np.ndarray:
    """
    Regner ut vektoren for den magnetiske feltstyrken
    ved punktet `r` fra en strømførende leder.

    Argumenter:
        r: np.ndarray
            Punkt i rommet
        koordinater: np.ndarray
            Liste med alle punkter på lederkurven
        i: float = I
            Strømmen gjennom lederen
    """

    # Start med en nullvektor for feltvektoren
    B = np.zeros(3)

    # Hent antall punkter i angitt lederkurve
    N = koordinater.shape[0]

    # Regn ut bidraget fra hvert punkt i lederkurven
    for n in range(N):
        # Regn ut det magnetiske feltet for en lukket sløyfe
        n0 = n
        n1 = (n + 1) % N  # Sikrer at vi går tilbake til start

        # Midtpunktet til segmentet
        mid = (koordinater[n1] + koordinater[n0]) / 2

        # Regn ut vektorer
        R_vec = r - mid  # Avstandsvektor
        dlv = koordinater[n1] - koordinater[n0]  # Linjeelement
        norm_R = np.linalg.norm(R_vec)  # Normalisert avstandsvektor

        # Regn ut differensialelementet for det magnetiske feltet
        # og regn ut netto magnetisk felt styrke ved å summere
        # opp differensialelementer
        dB = (mu0 * i / (4 * np.pi)) * np.cross(dlv, R_vec) / np.power(norm_R, 3)
        B += dB

    # Returner feltet
    return B