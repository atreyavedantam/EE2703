import numpy as np
import matplotlib.pyplot as plt


def gradDesc(guess: list, f, fgrad, ranges: list[list], a, iterations, tol):
    # implements the gradient descent
    guess = np.array(guess)
    guesses = [list(guess)]
    while True:
        gradient = np.array(fgrad(guess))
        guess = np.subtract(guess, a * gradient)  # gradient update

        # check if the new guess is within range
        for x in range(len(guess)):
            if guess[x] < ranges[x][0]:
                guess[x] = ranges[x][0]
            if guess[x] > ranges[x][1]:
                guess[x] = ranges[x][1]

        guesses.append(list(guess))

        if len(guesses) < 2:
            continue
        
        if f(guesses[-1]) == 0:
            continue  # relative error is not defined in this case - avoid division by zero error
        
        relError = np.abs(f(guesses[-1]) - f(guesses[-2]) / f(guesses[-1]))  # relative error for terminating loop

        if relError < tol:
            break

        iterations -= 1  # max number of iterations in case relError never subceeds tolerance
        if iterations == 0:
            break

    return guess, guesses  # the optimum and the path taken to reach the optimum


# plot a given function for better visualization
def plotFunc(func, range, resolution, min, dim, path):
    if dim == 1:

        x = np.linspace(range[0], range[1], resolution)  # define the independent parameter
        y = func(x)

        plt.title('Plot of the function and the gradient descent steps (in red)')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.plot(x, y)
        plt.plot(path, func(path), 'ro')  # plot the path of the gradient descent
        plt.show()

    elif dim == 2:
        # define the independent parameters
        x = np.linspace(range[0][0], range[0][1], resolution)
        y = np.linspace(range[1][0], range[1][1], resolution)

        X, Y = np.meshgrid(x, y)
        Z = func([X, Y])
        path = np.array(path)

        u = path[:, 0]
        v = path[:, 1]
        funcValues = []
        for point in path:
            funcValues.append(func(point))  # contains z-values of the gradient descent path

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
        ax.scatter(u, v, funcValues, facecolor='red')
        ax.set_title('Contour plot of the function with the gradient descent steps (in red)', fontsize=14)
        ax.set_xlabel('x-axis', fontsize=12)
        ax.set_ylabel('y-axis', fontsize=12)
        ax.set_zlabel('z-axis', fontsize=12)

        plt.show()
    return None


# functions for various problems
def f1(x):
    x = np.array(x)
    return x ** 2 + 3 * x + 8


def df1dx(x):
    x = np.array(x)
    return 2 * x + 3


def f3(params):
    x, y = params[0], params[1]
    return np.array(x ** 4 - 16 * x ** 3 + 96 * x ** 2 - 256 * x + y ** 2 - 4 * y + 262)


def df3_dx(x, y):
    return 4 * x ** 3 - 48 * x ** 2 + 192 * x - 256


def df3_dy(x, y):
    return 2 * y - 4


def f4(params):
    x, y = params[0], params[1]
    return np.exp(-(x - y) ** 2) * np.sin(y)


def df4_dx(x, y):
    return -2 * np.exp(-(x - y) ** 2) * np.sin(y) * (x - y)


def df4_dy(x, y):
    return np.exp(-(x - y) ** 2) * np.cos(y) + 2 * np.exp(-(x - y) ** 2) * np.sin(y) * (x - y)


# take in the individual partial derivatives and return the gradient
def f3grad(params):
    x, y = params[0], params[1]
    return df3_dx(x, y), df3_dy(x, y)


def f4grad(params):
    x, y = params[0], params[1]
    return df4_dx(x, y), df4_dy(x, y)


def f5(x):
    return np.cos(x) ** 4 - np.sin(x) ** 3 - 4 * np.sin(x) ** 2 + np.cos(x) + 1


def df5dx(x):  # manually calculated derivative for problem 4
    return -4 * np.sin(x) * np.cos(x) ** 3 - 3 * np.cos(x) * np.sin(x) ** 2 - 8 * np.sin(x) * np.cos(x) - np.sin(x)


# parameters for the whole algorithm
lr = 0.1
tolerance = 0.001
numSteps = 1e4

# parameters for problem 1
guess1 = [3]
lim1 = [-5, 5]

# parameters for problem 2
guess2 = [6, 2]
xlim2 = [-10, 10]
ylim2 = [-10, 10]

# parameters for problem 3
guess3 = [0, 0]
xlim3 = [-np.pi, np.pi]
ylim3 = [-5, 10]

# parameters for problem 4
guess4 = [3]
lim4 = [0, 2*np.pi]

sol1, path1 = gradDesc(guess1, f1, df1dx, [lim1], lr, numSteps, tolerance)
sol2, path2 = gradDesc(guess2, f3, f3grad, [xlim2, ylim2], lr, numSteps, tolerance)
sol3, path3 = gradDesc(guess3, f4, f4grad, [xlim3, ylim3], lr, numSteps, tolerance)
sol4, path4 = gradDesc(guess4, f5, df5dx, [lim4], lr, numSteps, tolerance)

print('1. The minimum occurs at x = ', sol1, 'and f(x) = ', f1(sol1))
print('2. The minimum occurs at x = ', sol2, 'and f(x, y) = ', f3(sol2))
print('3. The minimum occurs at x = ', sol3, 'and f(x, y) = ', f4(sol3))
print('4. The minimum occurs at x = ', sol4, 'and f(x) = ', f5(sol4))

resol = 1000
# choosing different range for plotting f2 alone for aesthetic appeal
xlim2plot = [0, 10]
ylim2plot = [0, 6]

# plots
plotFunc(f1, lim1, resol, sol1, 1, path1)
plotFunc(f3, [xlim2plot, ylim2plot], resol, sol2, 2, path2)
plotFunc(f4, [xlim3, ylim3], resol, sol3, 2, path3)
plotFunc(f5, lim4, resol, sol4, 1, path4)
