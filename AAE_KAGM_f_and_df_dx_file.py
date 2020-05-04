from normsdist_erf_file import *

import numpy



def AAE_KAGM_f_and_df_dx(x_t, rL, KappaQ2, Sigma1, Sigma2, Rho12, TauGrid, ZLB_Imposed):
    #UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    x_t = numpy.array(x_t)
    TauGrid = numpy.array(TauGrid)[:,0]

    g1 = numpy.ones(max(numpy.shape(TauGrid)))
    G1 = numpy.copy(TauGrid)
    g2 = numpy.exp(-KappaQ2*TauGrid)
    G2 = (1 - g2) / KappaQ2

    # Expected path of short rate.
    SR = x_t[0] * g1 + x_t[1] * g2

    # Volatility effect.
    VE = -0.5 * Sigma1**2 * numpy.multiply(G1,G1) - 0.5 * Sigma2**2 * numpy.multiply(G2,G2) - Rho12 * Sigma1 * Sigma2 * numpy.multiply(G1,G2)

    # Forward rate.
    GATSM_f = SR + VE

    if (ZLB_Imposed == 1):
        # Calculate annualized option volatility, Omega.
        G_11 = numpy.copy(TauGrid)
        G_22 = (1 - numpy.exp(-2*KappaQ2*TauGrid)) / (2 * KappaQ2)
        G_12 = numpy.copy(G2)
        Omega = numpy.sqrt(Sigma1**2 * G_11 + Sigma2**2 * G_22 + 2 * Rho12 * Sigma1 * Sigma2 * G_12)

        # Calculate cumulative normal probabilities for N(0,1) distribution.
        d = numpy.divide(GATSM_f-rL, Omega,where=Omega!=0)
        normsdist_erf_d = normsdist_erf(d)

        # Calculate gradiant and CAB_GATSM_f
        CAB_GATSM_df_dx = numpy.transpose(numpy.vstack((numpy.multiply(g1,normsdist_erf_d), numpy.multiply(g2,normsdist_erf_d))))
        CAB_GATSM_f = rL + numpy.multiply(GATSM_f-rL, normsdist_erf_d) + numpy.multiply(numpy.exp(-0.5*numpy.multiply(d,d)), Omega) / numpy.sqrt(2*numpy.pi)
    else:
        # ZLB not imposed, so just constant gradiant and GATSM_f.
        CAB_GATSM_df_dx = numpy.transpose(numpy.vstack((g1,g2)))
        CAB_GATSM_f = numpy.copy(GATSM_f)

    return (CAB_GATSM_f,CAB_GATSM_df_dx)



