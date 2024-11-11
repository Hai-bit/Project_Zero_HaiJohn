#!/usr/bin/env python3
# -*- code: utf-8 -*-

"""
En bibliotek som inneholder felles rutiner
og definisjoner som tas i bruk i numeriske
utregninger for Computational Essay i
FYS1120 - semester H 2024
"""


# Eksterne biblioteker
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange  # type: ignore
from typing import Optional


# Konstanter
mu0 = 4 * np.pi * 1.0e-7  # Permeabilitiet i vakuum (T * m / A)
I = 1.0  # Strømmen gjennom solenoiden (A)

# --- Metoder --- #

# Metode for å regne ut magnetisk feltstyrke
# fra en strømførende leder
# - bruker eksemplet fra pensum: Chapter 11.1
# "Elementary Electromagnetism Using Python"
# - optimalisert ved bruk av Numba
@njit
def bfieldlist(r: np.ndarray, koordinater: np.ndarray, i: float) -> np.ndarray:
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


# Metode for å regne ut magnetisk feltstyrke over
# et rutenett (grid)
# - optimalisert ved bruk av Numba (parallellkjøring)
@njit(parallel=True)
def beregn_B_felt(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, koordi: np.ndarray, i: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regner ut magnetisk feltstyrke over et rutenett (grid)

    Argumenter:
        X: np.ndarray
            Liste med x-koordinater
        Y: np.ndarray
            Liste med y-koordinater
        Z: np.ndarray
            Liste med z-koordinater
        koordi: np.ndarray
            Liste med punkter på lederkurven
        i: float = I
            Strøm gjennom lederen
    """

    # Initialiser tre lister for hvert
    # komponent i feltvektoren
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    # Regn ut feltstyrken for hvert punkt i planet
    for u in prange(X.shape[0]):
        for v in range(X.shape[1]):
            r = np.array([X[u, v], Y[u, v], Z[u, v]])
            B = bfieldlist(r, koordi, i)
            Bx[u, v] = B[0]
            By[u, v] = B[1]
            Bz[u, v] = B[2]

    return Bx, By, Bz


# Metode for å plotte magnetisk feltstyrke
def plottingsone(
    B1: np.ndarray,
    B2: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    navn: list[str],
    farge: str,
    R: Optional[float] = None,
    L: Optional[float] = None,
    tol: float = 1.0e-6,
) -> None:
    """
    Plotter magnetisk feltstyrke på et plan

    Argumenter:
        B1: np.ndarray
            Første komponent i feltvektoren
        B2: np.ndarray
            Andre komponent i feltvektoren
        axis1: np.ndarray
            Første akse i planet
        axis2: np.ndarray
            Andre akse i planet
        navn: str
            Navn til akser på figuren
        farge: str
            Farge til feltlinjene
        R: Optional[float] = None
            Solenoidens radius
        L: Optional[float] = None
            Solenoidens lengde
        tol: float = 1.0e-6
            Toleranse der verdier under den
            betraktes til å være tilnærmet
            lik null
    """

    # Lag en figur
    plt.figure(figsize=(8, 6))

    # Plott feltstyrken i et arealområde
    B_magnitude = np.sqrt(B1 ** 2 + B2 ** 2)
    contour = plt.contourf(axis1, axis2, B_magnitude, levels=50, cmap="viridis")
    cbar = plt.colorbar(contour, ax=plt.gca())
    cbar.set_label("Magnetfeltstyrke (T)")

    # Plott feltlinjene på planet
    plt.streamplot(axis1, axis2, B1, B2, color=farge, density=1.5)

    # Finn punktene der feltstyrken er lik null
    zero_field = B_magnitude <= tol  # Områder med felt under terskelverdien

    # Plott områdene med feltstyrke under terskelen
    plt.contourf(axis1, axis2, zero_field, levels=[tol, 1], colors="red", alpha=0.5)

    # Legg til fargesøyle
    cbar = plt.colorbar()
    cbar.set_label("$|B| = 0$")

    plt.xlabel(f"{navn[1]} (m)")
    plt.ylabel(f"{navn[2]} (m)")

    title = f"B-felt rundt en solenoide i {navn[0]}-planet"

    if (R) and (not L):
        title += f" ({R=})"
    elif (not R) and (L):
        title += f" ({L=})"
    elif R and L:
        title += f" ({R=}, {L=})"

    plt.title(title) #Added inn the title since it was not counted in
    plt.grid(True)
