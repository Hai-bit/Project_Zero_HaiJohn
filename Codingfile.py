import numpy as np
import matplotlib.pyplot as plt

#plasseres variabler 
# Konstanter
mu0 = 4 * np.pi * 1e-7  # Permeabiliteten i vakuum
I = 1.0  # Strøm gjennom solenoiden
R = 1.0  # Radius av solenoiden
N = 100  # Antall viklinger
L = 5.0  # Lengden på solenoiden
n = N / L  # Antall viklinger per lengdeenhet

# Definer grid
# Setter trapper hvor mye per område
step = 200
x = np.linspace(-5, 5, step)
z = np.linspace(-5, 5, step)
y = np.linspace(-5, 5, step)
X, Z = np.meshgrid(x, z)

# Beregn magnetfeltkomponentene
def magnetic_fieldXZ(x, z):
    #Lager liste med to sider 
    Bx = np.zeros_like(x)
    Bz = np.zeros_like(z)
    # Innvendig felt (approksimert som uniformt)
    inside = (np.abs(z) <= L/2) & (np.sqrt(x**2) <= R)
    Bz[inside] = mu0 * n * I  # Uniformt felt inne i solenoiden
    # Utvendig felt (forenklet modell)
    outside = ~inside
    r = np.sqrt(x[outside]**2 + z[outside]**2)
    Bx[outside] = mu0 * I * R**2 * x[outside] / (2 * (r**2 + R**2)**(1.5))
    Bz[outside] = mu0 * I * R**2 * z[outside] / (2 * (r**2 + R**2)**(1.5))
    return Bx, Bz

Bx, Bz = magnetic_fieldXZ(X, Z)

# Plot magnetfeltlinjene I xz planet
plt.streamplot(X, Z, Bx, Bz, color='b', density=1.5, linewidth=1, arrowsize=1)
# Tegn solenoiden
plt.fill_between([-R, R], -L/2, L/2, color='gray', alpha=0.3)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Magnetfeltlinjer rundt en solenoide')
plt.axis('equal')
plt.grid(True)
plt.show()
