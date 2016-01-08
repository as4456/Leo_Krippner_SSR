from AAE_KAGM_f_and_df_dx_file import *

import numpy
import numpy.matlib



def AAD_KAGM_R_and_dR_dx(x_t, rL, KappaQ2, Sigma1, Sigma2, Rho12, Tau_K, dTau, ZLB_Imposed):
    #UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    Tau_K_T = numpy.intp(numpy.round(Tau_K / dTau))

    TauMax = numpy.amax(Tau_K)
    TauGrid = numpy.matrix(numpy.arange(0, TauMax+dTau*1.0e-12, dTau)).getH()

    (CAB_GATSM_f, CAB_GATSM_df_dx) = AAE_KAGM_f_and_df_dx(x_t, rL, KappaQ2, Sigma1, Sigma2, Rho12, TauGrid, ZLB_Imposed)

    CAB_GATSM_R = numpy.matrix(numpy.cumsum(CAB_GATSM_f, axis=0)).getH()
    CAB_GATSM_R = numpy.divide(CAB_GATSM_R, numpy.matrix(numpy.arange(1,len(CAB_GATSM_f)+1)).getH())
    CAB_GATSM_dR_dx = numpy.cumsum(CAB_GATSM_df_dx, axis=0)
    CAB_GATSM_dR_dx = numpy.divide(CAB_GATSM_dR_dx, numpy.matlib.repmat(numpy.matrix(numpy.arange(1,max(numpy.shape(CAB_GATSM_df_dx))+1)).getH(),1,2))

    # Record required results in matrix. 
    CAB_GATSM_R = CAB_GATSM_R[Tau_K_T-1]
    CAB_GATSM_dR_dx = CAB_GATSM_dR_dx[Tau_K_T-1,:]

    return (CAB_GATSM_R,CAB_GATSM_dR_dx)


