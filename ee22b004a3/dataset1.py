import numpy as np
import matplotlib.pyplot as plt


def coordinateReturner(filename):
    # returns two lists containing the x and the y coordinates - useful for analysis

    fp = open(filename, 'r')
    file = fp.readlines()

    xc = []
    yc = []

    for coordinate in file:
        coordinate = coordinate.split()
        xc.append(float(coordinate[0]))
        yc.append(float(coordinate[1]))

    return xc, yc


def dataset1(filename):

    xc, yc = coordinateReturner(filename)

    plt.xlabel("X-coordinates")
    plt.ylabel("Y-coordinates")
    plt.title("Noisy data with a linear functional fit")
    plt.errorbar(xc[::25], yc[::25], 2, fmt='ro')

    M = np.column_stack([xc, [1 for i in range(len(yc))]])  # creation of the M matrix to minimise errors

    (p1, p2), _, _, _ = np.linalg.lstsq(M, yc, rcond=None)

    line = []

    for i in range(len(xc)):
        line.append(p1 * xc[i] + p2)  # will store all the y-coordinates of the linear regressor

    plt.plot(xc, yc, 'g', xc, line, 'm')
    plt.legend(['Noisy Data', 'Regression', 'Error Bars'], loc='lower right')
    plt.savefig("a3_dataset1fig1.png")

    return f"The best linear fit for this data is y = {p1}x + {p2}"


print(dataset1('dataset1.txt'))