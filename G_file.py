import numpy


def G(Phi, Tau):
    #UNTITLED2 Summary of this function goes here
    #   Detailed explanation goes here

    if (Phi <= 0):
        G_Phi_Tau = Tau
    else:
        invPhi = 1 / Phi
        PhiTau = Phi * Tau
        ExpNegPhiTau = numpy.exp(-PhiTau)
        G_Phi_Tau = invPhi * (1 - ExpNegPhiTau)


    return G_Phi_Tau


