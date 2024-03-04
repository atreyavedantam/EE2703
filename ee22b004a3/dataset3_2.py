import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def coordinateReturner(filename):
    # returns the coordinates as two separate lists for analysis

    fp = open(filename, 'r')
    file = fp.readlines()

    xc = []
    yc = []

    for coordinate in file:
        coordinate = coordinate.split()
        xc.append(float(coordinate[0]))
        yc.append(float(coordinate[1]))

    return xc, yc


def blackBodyRadiation(f, T, h, kb, c):
    # defines the Planck's equation for determining the intensity of the Black body radiation

    modifT = h/(kb*T)  # grouping all the constants - better readability
    denom = np.exp(modifT*f) - 1
    numer = 2*h*np.power(f, 3)/(c**2)
    return np.divide(numer, denom)


freq, intensity = coordinateReturner('dataset3.txt')
plt.plot(freq, intensity)

# converting the lists into numpy arrays for curve_fit
freq, intensity = np.array(freq), np.array(intensity)
(T, h, kb, c), _ = curve_fit(blackBodyRadiation, freq, intensity, [4997, 6.626e-34, 1.38e-23, 3e8])

plt.plot(freq, blackBodyRadiation(freq, T, h, kb, c))

plt.title('Intensity of Black Body radiation vs frequency and best Planck equation fit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Intensity of Black Body radiation')
plt.legend(['Noisy data', 'Predicted intensity as per Planck\'s equation'], loc='upper right')
plt.savefig("a3_dataset3fig2.png")

print(f"The estimated parameters are: T = {T} K\n h = {h} Js\n kb = {kb} J/K\n c = {c} m/s")