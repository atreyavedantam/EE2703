import numpy as np
import matplotlib.pyplot as plt


def travelDistance(cities, cityorder):
    # returns the travel distance between all the cities, given a certain order of travelling
    totaldistance = 0
    ogCity = np.array([cityorder[0]])
    cityorder = np.concatenate((cityorder, ogCity)) # append the first city to calculate the distance for the round trip

    for i in range(len(cityorder) - 1):
        city1 = cityorder[i]
        city2 = cityorder[i+1]
        totaldistance += distance(cities[city1][0], cities[city1][1], cities[city2][0], cities[city2][1])

    return totaldistance


def distance(x1, y1, x2, y2):
    # returns the distance between two cities given its coordinates
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def tsp(cities, stochasticity=30):
    # performs simulated annealing to optimize the travel path

    T = 1e9  # temperature for annealing
    k = 1 - 1e-4  # decay rate

    cityorder = np.arange(len(cities))
    np.random.shuffle(cityorder)  # start with a randomly shuffled order

    bestDistance = travelDistance(cities, cityorder)
    whileChecker = 1e6  # makes sure that the while loop breaks in case of some error - good programming practice

    while T > 1:
        i, j = np.random.choice(cityorder, 2, replace=False)  # choose two random cities

        modifiedorder = cityorder.copy()
        modifiedorder[i], modifiedorder[j] = modifiedorder[j], modifiedorder[i]

        if travelDistance(cities, modifiedorder) < bestDistance:
            # if the new travel distance is smaller, update the order and the best distance for sure
            bestDistance = travelDistance(cities, modifiedorder)
            cityorder = modifiedorder
        else:
            guess = np.random.random()
            probability = np.exp(stochasticity * (bestDistance - travelDistance(cities, modifiedorder))/T)
            # with a probability dependent on the magnitude of the difference, accept the modified order
            if guess < probability:
                bestDistance = travelDistance(cities, modifiedorder)
                cityorder = modifiedorder

        T *= k  # decay the temperature
        whileChecker -= 1

        if whileChecker == 0:
            print('While Loop Forcibly Terminated')
            break

    return cityorder


def relErrorCheck(cities, optimisedorder, initialorder):
    # given the initial order of cities and the optimal order of cities, gives the error
    err = np.abs(travelDistance(cities, optimisedorder) - travelDistance(cities, initialorder))/travelDistance(
        cities, initialorder)
    return f'{err*100}' + '%'


def fileParser(file):
    # reads the file and returns its contents in a list of 2-tuples (necessary format for input)
    fp = open(file, 'r')
    fileContents = fp.readlines()
    fileContents.pop(0)
    cities = []
    for content in fileContents:
        x, y = content.split()
        coordinates = (float(x), float(y))
        cities.append(coordinates)
    return cities


filename = 'tsp40.txt'
citynames = fileParser(filename)
optimalOrder = tsp(citynames, stochasticity=200)
error = relErrorCheck(citynames, optimalOrder, np.arange(len(citynames)))

n = len(citynames)

print('The optimized order is', optimalOrder)
print('The distance covered is: ', travelDistance(citynames, optimalOrder), 'with an improvement of ', error)

x, y = [], []
for city in optimalOrder:
    # plot all the cities and append the path to two lists to plot them as well with a different colour
    plt.plot(citynames[city][0], citynames[city][1], 'g')
    plt.annotate(f"Point {city}", (citynames[city][0], citynames[city][1]))
    x.append(citynames[city][0])
    y.append(citynames[city][1])

# plotting the path
plt.plot(x, y, 'ro')
plt.plot(x, y, 'b')
plt.title('Plot of the optimized path')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
