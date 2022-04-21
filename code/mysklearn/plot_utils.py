
import matplotlib.pyplot as plt

import mysklearn.utils as utils

def create_bar_chart(title, xlabel, ylabel, values, counts):
    plt.figure(figsize=(30,10))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(values, counts, align="center", edgecolor="black")
    plt.show()

def creat_pie_chart(x_val, y_val, title):
    plt.figure()
    plt.title(title)
    plt.pie(y_val, labels = x_val, autopct="%1.1f%%")
    plt.show()

def discretization_bar_chart(freqs, cutoffs, title, xlabel, ylabel):
    plt.figure(figsize=(30,10))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(cutoffs[:-1], freqs, width=(cutoffs[1] - cutoffs[0]),
    edgecolor="black", align="edge")

def create_histogram(title, xlabel, ylabel, values):
    plt.figure()
    plt.hist(values, bins=10) # default is 10
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def compute_slope(xdata, ydata):
    xmean = sum(xdata)/len(xdata)
    ymean = sum(ydata)/len(ydata)
    slope = sum([(xdata[i] - xmean) * (ydata[i] - ymean) for i in range(len(xdata))])\
         / sum([(xdata[i] - xmean) ** 2 for i in range(len(xdata))])
    b_int = ymean - slope * xmean
    return slope, b_int

def create_scatter_plot(title, xlabel, ylabel, xdata, ydata):
    plt.figure()
    plt.scatter(xdata, ydata)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    coefficient, covariance = utils.compute_covariance_and_coefficient(xdata, ydata)
    slope, b_int = compute_slope(xdata, ydata)
    plt.annotate("Coefficient = " + str(round(coefficient, 5)), xy=(.5,.8), \
        xycoords="axes fraction", horizontalalignment="center")
    plt.annotate("Covariance = " + str(round(covariance, 5)), xy=(.5,.7), \
        xycoords="axes fraction", horizontalalignment="center")

    plt.plot([min(xdata), max(xdata)], [slope * min(xdata) + b_int, slope\
         * max(xdata) + b_int], c="r")

    plt.show()