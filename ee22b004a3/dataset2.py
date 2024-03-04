import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin
from scipy.optimize import curve_fit


def coordinateReturner(filename):
    # returns two lists containing x and y coordinates separately - useful for later analysis

    fp = open(filename, 'r')
    file = fp.readlines()

    xc = []
    yc = []
    for coordinate in file:
        coordinate = coordinate.split()
        xc.append(float(coordinate[0]))
        yc.append(float(coordinate[1]))
    return xc, yc


def sinMaker(inputs, frequency):
    # takes the input coordinates array and returns a sinusoid with the given frequency

    sineArray = []
    for i in range(len(inputs)):
        sineArray.append(sin(2*pi*frequency*inputs[i]))
    return sineArray


def lstsqFit(filename, f):
    # uses numpy.linalg.lstsq to fit three sine waves at frequencies f, 3f and 5f

    xc, yc = coordinateReturner(filename)
    M = np.column_stack([sinMaker(xc, f), sinMaker(xc, 3*f), sinMaker(xc, 5*f)])
    (p1, p2, p3), _, _, _ = np.linalg.lstsq(M, yc, rcond=None)

    line = []
    for i in range(len(xc)):
        line.append(p1 * sinMaker(xc, 0.4)[i] + p2 * sinMaker(xc, 1.2)[i] + p3 * sinMaker(xc, 2)[i])

    line = np.array(line)
    yc = np.array(yc)
    error = np.subtract(line, yc)

    # obtaining the total squared error for this fit - can compare fits with different frequencies using this
    error = error**2

    return p1, p2, p3, sum(error)


errorFrequency = []  # list of all the errors for 20 frequencies ranging from 0 to 2 Hz

for freq in range(0, 20):
    freq /= 10
    _, _, _, err = lstsqFit('dataset2.txt', freq)
    errorFrequency.append(err)

fr = errorFrequency.index(min(errorFrequency))/10  # best possible fundamental frequency - turns out to be 0.4

p1, p2, p3, _ = lstsqFit('dataset2.txt', fr)
xcor, ycor = coordinateReturner(filename='dataset2.txt')


def sineFit(xcoordinates, p1, p2, p3):
    # this function generates the sine curve for a frequency of "fr" (found to be 0.4) - need this for curve_fit

    sinF = np.array(sinMaker(xcoordinates, fr))
    sin3F = np.array(sinMaker(xcoordinates, 3*fr))
    sin5F = np.array(sinMaker(xcoordinates, 5*fr))
    return p1*sinF + p2*sin3F + p3*sin5F


def plotFit(p1, p2, p3, sp1, sp2, sp3, xc, yc):
    # plots the original noisy data, the fit using numpy.linalg.lstsq and the fit using scipy.optimize.curve_fit

    line1 = []
    line2 = []

    for i in range(len(xc)):
        line1.append(p1 * sinMaker(xc, 0.4)[i] + p2 * sinMaker(xc, 1.2)[i] + p3 * sinMaker(xc, 2)[i])
        line2.append(sp1 * sinMaker(xc, 0.4)[i] + sp2 * sinMaker(xc, 1.2)[i] + sp3 * sinMaker(xc, 2)[i])

    plt.plot(xc, yc, 'm', xc, line1, xc, line2)
    plt.legend(['Noisy data', 'Fit without curve_fit', 'Fit with curve_fit'], loc='lower right')
    plt.title('Curve fit of a sinusoid with three frequencies')
    plt.xlabel('X-coordinates')
    plt.ylabel('Y-coordinates')
    plt.savefig("a3_dataset2fig1.png")
    return 0


(sp1, sp2, sp3), _ = curve_fit(sineFit, xcor, ycor)
plotFit(p1, p2, p3, sp1, sp2, sp3, xcor, ycor)
print(f"The estimated curve (using np.linalg.lstsq) is y = {p1}sin({2*pi*fr}x) + {p2}sin({6*pi*fr}x) + {p3}sin({10*pi*fr}x)")
print(f"The estimated curve (using scipy.optimize.curve_fit) is y = {sp1}sin({2*pi*fr}x) + {sp2}sin({6*pi*fr}x) + {sp3}sin({10*pi*fr}x)")