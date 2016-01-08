from scipy import special
import numpy


def normsdist_erf(x):
    #UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    normsdist = 0.5 * (1 + special.erf(x / numpy.sqrt(2)))

    return normsdist



