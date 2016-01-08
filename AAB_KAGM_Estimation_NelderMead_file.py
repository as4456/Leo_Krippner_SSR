from AAC_KAGM_SingleLoop_file import *

import numpy
import scipy


def AAB_KAGM_Estimation_NelderMead(R_data, Tau_K, N, InitialParameters, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL, Iterations):
    #UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    Parameters = numpy.copy(InitialParameters)
    EKF_function = lambda Parameters: AAC_KAGM_SingleLoop(R_data, Tau_K, N, Parameters, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)[0]
    (FinalParameters,Fval) = scipy.optimize.fmin(func=EKF_function, x0=Parameters, xtol=float('Inf'), ftol=1e-2, maxiter=Iterations)

    return (FinalParameters,Fval)


