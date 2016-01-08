import numpy


def AAH_EMS_N2_function(Phi, x_T):
    (tmp,T) = numpy.shape(x_T)
    ES_CAB = numpy.ones((T,1)) * float('nan')
    EMS = numpy.ones((T,1)) * float('nan')
    ETZ = numpy.ones((T,1)) * float('nan')
    SSR = numpy.sum(x_T,axis=0)
    for t in range(0, T):
        x_t = x_T[:,t]
        if (SSR[t] >= 0):
            EMS_t = -x_t[1] / Phi
        else:
            Tau0 = -numpy.log(-x_t[0] / x_t[1]) / Phi
            ETZ[t] = Tau0
            EMS_t = x_t[0] * Tau0 - x_t[1] * numpy.exp(-Phi*Tau0) / Phi
        EMS[t] = EMS_t

    return (SSR, EMS, ETZ)



def AAH_EMS_N3_function(Phi, x_T, dTau):
    # Hyperparameters.
    dTau2 = 0.1 * dTau

    (tmp,T) = numpy.shape(x_T)
    TauGrid = numpy.matrix(numpy.arange(0, 100+1, dTau)).getH()
    TauGrid2 = numpy.matrix(numpy.arange(0, 1000+1, dTau2)).getH()

    EMS = numpy.ones((T,1)) * float('nan')
    EMS_TauH = numpy.ones((T,1)) * float('nan')
    TauH = 1
    ETZ = numpy.ones((T,1)) * float('nan')
    SSR = numpy.sum(x_T[0:1,:],axis=0)
    for t in range(0, T):
        x_t = x_T[:,t]
        E_Shadow_r = x_t[0] + x_t[1] * exp(-Phi*TauGrid) + x_t[2] * Phi * numpy.multiply(TauGrid, numpy.exp(-Phi*TauGrid))
        if (numpy.any(E_Shadow_r<0)):
            tmp1 = numpy.multiply(E_Shadow_r[0:-1], E_Shadow_r[1:])
            RootIndicator = numpy.amax(numpy.where(tmp1<0)[0] ,axis=0)
            tmp_func = lambda Tau0: x_t[0] + x_t[1] * exp(-Phi*Tau0) + x_t[2] * Phi * numpy.multiply(Tau0, exp(-Phi*Tau0))
            Tau0 = scipy.optimize.fsolve(tmp_func, numpy.matrix([TauGrid[RootIndicator],TauGrid[RootIndicator+1]]))
            E_Shadow_r = x_t[0] + x_t[1] * exp(-Phi*TauGrid2) + x_t[2] * Phi * numpy.multiply(TauGrid2, exp(-Phi*TauGrid2))
            E_CAB_r = numpy.maximum(E_Shadow_r, 0)
            dEMS = x_t[0] - E_CAB_r
            EMS_t = numpy.sum(dEMS,axis=0) * dTau2
            EMS[t] = EMS_t
            ETZ[t] = Tau0
            EMS_TauH_t = numpy.sum(dEMS[0:TauH/dTau2],axis=0) * dTau2
            EMS_TauH[t] = EMS_TauH_t
        else:
            EMS_t = -(x_t[1] + x_t[2]) / Phi
            EMS[t] = EMS_t
            EMS_TauH_t = -(x_t[1] + x_t[2]) * (1 - exp(-Phi*TauH)) / Phi + x_t[2] * exp(-Phi*TauH)
            EMS_TauH[t] = EMS_TauH_t

    return (SSR, EMS, ETZ)



def AAH_EMS_N23_function(Phi, x_T, dTau):
    (N,tmp) = numpy.shape(x_T)

    if (N == 2):
        (SSR, EMS, ETZ) = AAH_EMS_N2_function(Phi, x_T)
    else:
        (SSR, EMS, ETZ) = AAH_EMS_N3_function(Phi, x_T, dTau)

    return (SSR, EMS, ETZ)



