import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# defining the constants to be used
PLANCKS_CONSTANT = 6.62607015e-34
BOLTZMANN_CONSTANT = 1.380649e-23
SPEED_LIGHT = 299792458


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


def blackBodyRadiation(f, T):
    # defines the Planck's equation for determining the intensity of the Black body radiation

    modifT = PLANCKS_CONSTANT/(BOLTZMANN_CONSTANT*T)  # grouping all the constants - better readability
    denom = np.exp(modifT*f) - 1
    numer = 2*PLANCKS_CONSTANT*np.power(f, 3)/(SPEED_LIGHT**2)
    return np.divide(numer, denom)


freq, intensity = coordinateReturner('dataset3.txt')
plt.plot(freq, intensity)

# converting the lists into numpy arrays for curve_fit
freq, intensity = np.array(freq), np.array(intensity)
(T,), _ = curve_fit(blackBodyRadiation, freq, intensity, [500])

plt.plot(freq, blackBodyRadiation(freq, T))

plt.title('Intensity of Black Body radiation vs frequency and best Planck equation fit')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Intensity of Black Body radiation')
plt.legend(['Noisy data', 'Predicted intensity as per Planck\'s equation'], loc='upper right')
plt.savefig("a3_dataset3fig1.png")

print(f"The temperature at which this data was measured is estimated to be {T} Kelvin.")