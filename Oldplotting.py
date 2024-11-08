import numpy as np
import matplotlib.pyplot as plt

"""
THis python file is form old python code without using the 
bfield list, but due to some diifficulties form the XY plane
it was scrapped for sake for using the new code form the 
pensum -Hai Duc
"""
# Konstanter
mu0 = 4 * np.pi * 1e-7  # Permeabiliteten i vakuum
I = 1.0  # Strøm gjennom solenoiden
R = 1.0  # Radius av solenoiden
N = 100  # Antall viklinger
L = 5.0  # Lengden på solenoiden
n = N / L  # Antall viklinger per lengdeenhet

# Definer grid
step = 500
x = np.linspace(-4, 4, step)
y = np.linspace(-4, 4, step)
z = np.linspace(-4, 4, step)

# Planene
X, Z = np.meshgrid(x, z)
Y, Z_ = np.meshgrid(y, z)
X_, Y_ = np.meshgrid(x, y)

# Funksjoner for magnetfelt i de ulike planene med XZ og YZ
def magnetic_field(vec1, vec2):
    B1 = np.zeros_like(vec1)
    B2 = np.zeros_like(vec2)
    # Innvendig felt (approksimert som uniformt)
    inside = (np.abs(vec2) <= L/2) & (np.sqrt(vec1**2) <= R)
    B2[inside] = mu0 * n * I  # Uniformt felt inne i solenoiden
    # Utvendig felt (forenklet modell)
    outside = ~inside
    r = np.sqrt(vec1[outside]**2 + vec2[outside]**2)
    B1[outside] = mu0 * I * R**2 * vec1[outside] / (2 * (r**2 + R**2)**(1.5))
    B2[outside] = mu0 * I * R**2 * vec2[outside] / (2 * (r**2 + R**2)**(1.5))
    return B1, B2

def magnetic_fieldXY(x, y):
    Bx = np.zeros_like(x)
    By = np.zeros_like(y)
    Bz = np.zeros_like(x)
    # Innvendig felt (approksimert som uniformt)
    inside = (np.sqrt(x**2 + y**2) <= R)
    Bz[inside] = mu0 * n * I  # Uniformt felt inne i solenoiden
    # Utvendig felt (forenklet modell)
    outside = ~inside
    r = np.sqrt(x[outside]**2 + y[outside]**2)
    Bx[outside] = mu0 * I * R**2 * x[outside] / (2 * (r**2 + R**2)**(1.5))
    By[outside] = mu0 * I * R**2 * y[outside] / (2 * (r**2 + R**2)**(1.5))
    return Bx, By, Bz

# Hele plotter systemet
def plottingsone(B1, B2, axis1, axis2, name, farge):
    #Det for at den lager ny plott hele tiden den blir "kaldt opp"
    plt.figure(figsize=(8, 6))
    #plotter "bølgevektorene" 
    plt.streamplot(axis1, axis2, B1, B2, color=farge, density=1.5)
    
    # Beregn størrelsen på magnetfeltet
    B_magnitude = np.sqrt(B1**2 + B2**2)
    errorsone = 1e-9      # Definer en terskelverdi når tilnærmet 0
    zero_field = B_magnitude <= errorsone # Finner områder der magnetfeltet er under terskelverdien
    # plotter ut null område i plotten med lilla som viser 0 felt
    plt.contourf(axis1, axis2, zero_field, levels=[1e-9, 1], colors='purple', alpha=0.5)

    #plassere navn til 0 feltet
    cbar = plt.colorbar()
    cbar.set_label('0=B-felt')

    plt.xlabel(f'{name[1]} (m)')
    plt.ylabel(f'{name[2]} (m)')
    plt.title(f'B-felt rundt en solenoide i {name[0]}-planet')
    plt.grid(True)

# Plot i XZ-planet med blå farge
Bx, Bz = magnetic_field(X, Z)
namesoneXZ = ["XZ", "x", "z"]
plottingsone(Bx, Bz, X, Z, namesoneXZ, "b")
# Tegner solenoiden er
plt.fill_between([-R, R], -L/2, L/2, color='gray', alpha=0.3)

# Plot i YZ-planet med grønn farge
By, Bz = magnetic_field(Y, Z_)
namesoneYZ = ["YZ", "y", "z"]
plottingsone(By, Bz, Y, Z_, namesoneYZ, "g")
# Tegner solenoiden er
plt.fill_between([-R, R], -L/2, L/2, color='gray', alpha=0.3)

# Plot i XY-planet med rød farge
Bx, By, Bz = magnetic_fieldXY(X_, Y_)
namesoneXY = ["XY", "x", "y"]
plottingsone(Bx, By, X_, Y_, namesoneXY, "r")
# Tegner solenoiden er
circle = plt.Circle((0, 0), R, color='gray', alpha=0.3)
plt.gca().add_artist(circle)

plt.show()
