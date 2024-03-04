import numpy as np
import matplotlib.pyplot as plt


# exploratory data analysis
def partialDependencePlot(parameter, gre, toefl, rating, sop, lor, cgpa, research, chance):
    # plots the dependence of the chance of admission as a function of the given parameter, keeping the others constant
    params = list([gre, toefl, rating, sop, lor, cgpa, research, chance])
    chances = chance

    # take the average defined value of all quantities to use in plot
    pdef = [np.mean(gre), np.mean(toefl), np.mean(rating), np.mean(sop),
            np.mean(lor), np.mean(cgpa), np.mean(research)]
    plotArray = 0

    index = 0
    # plotArray will contain the x-variable; this will be decided based on what parameter is
    for i in range(len(params) - 1):
        if parameter != f'params[{i}]':
            params[i] = pdef[i]
        else:
            plotArray = params[i]
            index = i

    # plot the chance of admission against the parameter of interest
    plt.plot(plotArray, chances, 'o')
    plt.title("Chance of Admission versus " + varDict[index])
    plt.xlabel(varDict[index])
    plt.ylabel('Chance of Admission into a University')
    plt.show()
    return True


# finding the least square fit (linear model)
def findWeights(gre, toefl, rating, sop, lor, cgpa, research, chance, onlyFiveStar=0):
    # finds the appropriate weights of all the parameters to calculate the chance
    if onlyFiveStar == 0:
        M = np.column_stack([gre, toefl, rating, sop, lor, cgpa, research, [1 for i in range(len(gre))]])
    else:
        M = np.column_stack([gre, toefl, sop, lor, cgpa, research, [1 for i in range(len(gre))]])
    weights, _, _, _ = np.linalg.lstsq(M, chance, rcond=None)

    return weights


def fileParser(filename, onlyFiveStar=0):
    # parses the file and returns all the individual parameters as lists
    fp = open(filename, 'r')
    gre, toefl, rating, sop, lor, cgpa, research, chance = np.empty(0), np.empty(0), np.empty(0), np.empty(0), \
        np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    file = fp.readlines()
    file.pop(0)  # first line needs to be removed

    for line in file:

        components = line.split(',')
        if onlyFiveStar == 1 and components[3] != '5':  # if onlyFiveStar is 1, then only accept 5-rated universities
            continue

        gre = np.append(gre, components[1])
        toefl = np.append(toefl, components[2])
        rating = np.append(rating, components[3])
        sop = np.append(sop, components[4])
        lor = np.append(lor, components[5])
        cgpa = np.append(cgpa, components[6])
        research = np.append(research, components[7])
        chance = np.append(chance, components[8])

    # convert all lists to float datatype
    gre = gre.astype(float)
    toefl = toefl.astype(float)
    rating = rating.astype(float)
    sop = sop.astype(float)
    lor = lor.astype(float)
    cgpa = cgpa.astype(float)
    research = research.astype(float)
    chance = chance.astype(float)

    return gre, toefl, rating, sop, lor, cgpa, research, chance


def predictor(weights, parameters):
    # predicts a new chance given the weights
    probability = 0
    numParams = len(parameters)
    for i in range(numParams):
        probability += weights[i] * parameters[i]
    return probability + weights[numParams]


def errorLstsq(filename, weights):
    # gives the error in the linear regression model and the variance inherent in the data
    gre, toefl, rating, sop, lor, cgpa, research, chance = fileParser(filename)
    err = 0
    var = 0
    for i in range(len(chance)):
        err += (predictor(weights, [gre[i], toefl[i], rating[i], sop[i], lor[i], cgpa[i], research[i]]) - chance[i])**2
        var += (chance[i] - np.mean(chance))**2
    return err, var


filepath = "Admission_Predict_Ver1.1.csv"
varDict = {0: 'GRE', 1: 'TOEFL', 2: 'Rating', 3: 'SOP', 4: 'LOR', 5: 'CGPA', 6: 'Research', 7: 'Chance'}

g, t, rat, s, l, cg, res, ch = fileParser(filepath)

# partialDependencePlot("params[3]", g, t, rat, s, l, cg, res, ch)  # test for all parameters

w = findWeights(g, t, rat, s, l, cg, res, ch)
error, variance = errorLstsq(filepath, w)
r = np.sqrt(1-error/variance)  # r is an indicator of whether a linear model is a good fit for this data

print("The coefficient of determination is given by:", r)
if r > 0.82:
    print("The data is well explained by a linear relationship.")
elif r > 0.49:
    print("The data is reasonably explained by a linear fit but some non-linearity may be necessary.")
elif r > 0.17:
    print("The data might not be well explained a linear fit.")
else:
    print("The data does not have any semblance of linearity whatsoever.")

g, t, rat, s, l, cg, res, ch = fileParser(filepath, onlyFiveStar=1)

# partialDependencePlot("params[6]", g, t, rat, s, l, cg, res, ch)  # plot for all parameters to see dependence

w = findWeights(g, t, rat, s, l, cg, res, ch, onlyFiveStar=1)

# we print the max index - 1 (since rating doesn't exist anymore)
print("The most influencing factor in the admission for a top-rated university is", varDict[np.argmax(np.abs(w))-1])