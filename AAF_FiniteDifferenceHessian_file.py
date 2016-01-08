import numpy


def AAF_FiniteDifferenceHessian(Function, x, minDiff, R_data, Tau_K, N, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL):
    # FINDIFFHESSIAN calculates the numerical Hessian of funfcn evaluated at
    # the parameter set x using finite differences.
    # Adapted from MatLab code in fminusub, line 431-548.

    numberOfVariables = len(x)
    f = Function(R_data, Tau_K, N, x, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)[0]
    Hessian = numpy.zeros((numberOfVariables, numberOfVariables))

    # Define stepsize
    eps = numpy.finfo(1.).eps
    CHG = eps**(1.0/4) * numpy.multiply(numpy.sign(x), numpy.amax(numpy.abs(x),axis=0))
    # Make sure step size lies within DiffMinChange and DiffMaxChange
    CHG = numpy.multiply(numpy.sign(CHG+eps), numpy.amax(numpy.abs(CHG),axis=minDiff))

    fplus_array = numpy.zeros(numberOfVariables)

    for j in range(0, numberOfVariables):
        xplus = numpy.copy(x)
        xplus[j] = x[j] + CHG[j]
        # evaluate
        #(R_data,Lambda,ParamFix,x_Fixed,ParamFree,x_Free,Dt,Tau,EA)
        fplus = Function(R_data, Tau_K, N, xplus, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)[0]
        fplus_array[j] = fplus

    for i in range(0, numberOfVariables):
        # For each row, calculate the 2nd term in (4). This term is common to
        # the whole row and thus it can be reused within the current row: we
        # store it in fplus_i.
        xplus = numpy.copy(x)
        xplus[i] = x[i] + CHG[i]
        # evaluate  
        fplus_i = Function(R_data, Tau_K, N, xplus, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)[0]

        for j in range(i, numberOfVariables):   # start from i: only upper triangle
            # Calculate the 1st term in (2); this term is unique for each element
            # of Hessian and thus it cannot be reused.
            xplus = numpy.copy(x)
            xplus[i] = x[i] + CHG[i]
            xplus[j] = xplus[j] + CHG[j]
            # evaluate  
            fplus = Function(R_data, Tau_K, N, xplus, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)[0]
            Hessian[i,j] = (fplus - fplus_i - fplus_array[j] + f) / (CHG[i] * CHG[j])

    # Fill in the lower triangle of the Hessian
    Hessian = Hessian + numpy.triu(Hessian, k=1).getH()

    return Hessian



