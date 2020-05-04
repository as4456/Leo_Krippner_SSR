from globl import *
from G_file import *
from AAD_KAGM_R_and_dR_dx_file import *

import numpy
#import datetime
#import math
import scipy
import scipy.linalg
import matplotlib.pyplot as pyplot


def AAC_KAGM_SingleLoop(R_data, Tau_K, N, Parameters, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL):
    global Max_IEKF_Count
    global Max_IEKF_Point

    # Extended Kalman filter for the CAB-AFNSM(2).
    if (FINAL == 1):
        # The parameters are in their natural form, including Rho values.
        rL = Parameters[0]
        KappaQ2 = Parameters[1]
        KappaP = numpy.matrix([[Parameters[2], Parameters[3]], [Parameters[4], Parameters[5]]])
        ThetaP = numpy.matrix([[Parameters[6]], [Parameters[7]]])
        Sigma1 = abs(Parameters[8])
        Sigma2 = abs(Parameters[9])
        Rho12 = Parameters[10]
    else:
        # The parameters are in one of their restricted forms.
        if (KappaP_Constraint == 'Direct'):
            rL = Parameters[0]
            KappaQ2 = Parameters[1]
            KappaP = numpy.matrix([[Parameters[2], Parameters[3]], [Parameters[4], Parameters[5]]])
            ThetaP = numpy.matrix([[Parameters[6]], [Parameters[7]]])
            Sigma1 = abs(Parameters[8])
            Sigma2 = abs(Parameters[9])
            Rho12 = Parameters[10] / (1 + abs(Parameters[10]))
        else:
            KappaQ2 = Parameters[0]
            L = numpy.matrix([[Parameters[1], 0], [Parameters[2], Parameters[3]]])
            A = numpy.matrix([[0, Parameters[4]], [-Parameters[4], 0]])
            KappaP = numpy.matrix(L * L.getH() + A)
            ThetaP = numpy.matrix([[Parameters[5]], [Parameters[6]]])
            Sigma1 = abs(Parameters[7])
            Sigma2 = abs(Parameters[8])
            Rho12 = Parameters[9] / (1 + abs(Parameters[9]))
    #

    SIGMA = numpy.matrix([[Sigma1, 0], [Rho12*Sigma2, Sigma2*numpy.sqrt(1-Rho12**2)]])
    OMEGA = SIGMA * SIGMA.getH()
    Sigma_Nu = Parameters[11:]

    (T,K) = numpy.shape(R_data)
    KT = K * T

    # Extended Kalman filter items.
    x_T = numpy.ones((N,T)) * float('nan')
    P_T = numpy.ones((N,N,T)) * float('nan')

    # Calculate the state equation quantities based on parameter values.
    D, V = numpy.linalg.eig(KappaP)
    #D = numpy.transpose(numpy.matrix(D))
    d1 = D[0]
    d2 = D[1]
    if ((d1.real < 0) or (d2.real < 0)):
        if (d1.real < 0):
            d1 = 1e-6 + d1.imag * 1j
        if (d2.real < 0):
            d1 = 1e-6 + d2.imag * 1j

        D = numpy.matrix(numpy.diag([d1, d2]))
        KappaP = numpy.real(numpy.linalg.solve(V.getH(), (V*D).getH()).getH())

    F = numpy.matrix(scipy.linalg.expm(-KappaP*Dt))
    D, V = numpy.linalg.eig(F)
    if (numpy.any(numpy.abs(D)>1.0001)):
        print('SingleLoop:argChk   abs(eig(F))>1')
        exit()

    Q = numpy.matrix([[G(2*d1,Dt), G(d1+d2,Dt)], [0, G(2*d2,Dt)]])
    Q = Q + numpy.transpose(numpy.triu(Q, k=1))
    # NOTE: Use TRANSPOSE, because ' gives congugate transpose.
    tmp1 = numpy.transpose(V)
    tmp2 = numpy.matrix(numpy.linalg.solve(V, OMEGA))
    U = numpy.linalg.solve(tmp1.getH(), tmp2.getH()).getH()
    Q = numpy.multiply(U, Q)
    Q = V * Q * numpy.transpose(V)
    Q = numpy.real(Q)

    # The measurement equation quantities depend on the state variables, and
    # so have to be re-calculated at each step of  the Kalman filter.

    # Starting the Recursion.
    # Unconditional mean and variance of state variable vector.
    x_Plus = numpy.copy(ThetaP)
    P_Plus = numpy.matrix([[0.5/d1, 1/(d1+d2)], [0, 0.5/d2]])
    P_Plus = P_Plus + numpy.transpose(numpy.triu(P_Plus, k=1))
    P_Plus = numpy.multiply(U, P_Plus)
    P_Plus = V * P_Plus * numpy.transpose(V)
    P_Plus = numpy.real(P_Plus)

    #IEKF_Count=2;
    logL = 0
    # Extended Kalman filter if IEKF_Count=0, iterated EKF with fixed
    # IEKF_Count iterations if IEKF_Count>0, and iterated EKF with fixed
    # tolerance abs(IEKF_Count) if IEKF_Count<0.
    if (IEKF_Count < 0):
        x_Tolerance = abs(IEKF_Count)
        IEKF_Count = 20

    for t in range(0, T):
        # Forecast step.
        x_Minus = (numpy.eye(N) - F) * ThetaP + F * x_Plus
        P_Minus = F * P_Plus * F.getH() + Q

        # Update step.
        # Observations for time t.
        y_Obs = numpy.matrix(0.01 * R_data[t,:]).getH()
        y_Missing = numpy.squeeze(numpy.array(numpy.isnan(y_Obs))) #numpy.array(numpy.isnan(y_Obs))#[:,0]
        y_Obs = y_Obs[~y_Missing]
        R = numpy.diag(numpy.power(Sigma_Nu[~y_Missing], 2))

        x_Plus_i_Minus_1 = numpy.copy(x_Minus)
        x_Plus_i0 = numpy.copy(x_Minus)
        for i in range(1, 1+IEKF_Count+1):
            # EKF and IEKF iterations
            # Following Simon (2006), p. 409 and pp. 411-12.
            # Note that the EKF step is the i=1 iteration. To see this, note
            # that x_Minus - x_Plus_i0 = 0 in eq. 13.64.
            # y_t_Hat=h(x_Minus,0), i.e. fitted values of y_t given x_Minus.
            # Ht=dR/dX(x_Minus), i.e. the Jacobian given x_Minus.

            (y_Hat, H_i) = AAD_KAGM_R_and_dR_dx(x_Plus_i0, rL, KappaQ2, Sigma1, Sigma2, Rho12, Tau_K, dTau, ZLB_Imposed)

            y_Hat = y_Hat[~y_Missing]
            H_i = H_i[~y_Missing,:]
            HPHR_i = (H_i * P_Minus * H_i.getH() + R)
            K_i = numpy.linalg.solve(HPHR_i.getH(), (P_Minus*H_i.getH()).getH()).getH()
            w_i = y_Obs - y_Hat - H_i * (x_Minus - x_Plus_i0)
            x_Plus_i1 = x_Minus + K_i * w_i

            if (IEKF_Count == 20):
                # Using tolerance, so check for convergence.
                if (i > 15):
                    # Large number of iterations, so print output to screen.
                    print(t, i-1, numpy.matrix(x_Plus_i1).getH(), numpy.matrix(x_Plus_i0).getH(), numpy.matrix(x_Plus_i1).getH() - numpy.matrix(x_Plus_i0).getH())
                if (numpy.all(numpy.abs(x_Plus_i1-x_Plus_i0)<x_Tolerance)):
                    # Difference from last update within tolerance, so exit.
                    break
                if (numpy.all(numpy.abs(x_Plus_i1-x_Plus_i_Minus_1)<x_Tolerance)):
                    # Allows for numerical cycling between i+1, i, i-1 updates.
                    # Difference from i-1 update within tolerance, so exit.
                    x_Plus_i1 = 0.5 * (x_Plus_i1 + x_Plus_i0)
                    break

            # Record these values to allow testing for convergence.
            x_Plus_i_Minus_1 = numpy.copy(x_Plus_i0)
            x_Plus_i0 = numpy.copy(x_Plus_i1)

        # Calculate final posterior values and record values.
        x_Plus = numpy.copy(x_Plus_i1)
        P_Plus = (numpy.matrix(numpy.eye(N)) - K_i * H_i) * P_Minus
        x_T[:,t] = x_Plus[:,0]
        P_T[:,:,t] = numpy.copy(P_Plus)
        logL = logL + numpy.log(numpy.linalg.det(HPHR_i), where=numpy.linalg.det(HPHR_i)>0) + numpy.linalg.solve(HPHR_i.getH(),w_i).getH() * w_i

        # Hold IEKF count.
        if Max_IEKF_Count is None:
            Max_IEKF_Count = IEKF_Count
        if (i-1 > Max_IEKF_Count):
            Max_IEKF_Count = i - 1
            Max_IEKF_Point = t

        # disp([num2str(t),' ',num2str(i-1),' ',num2str(x_Plus_i1'-x_Plus_i0')])
        # format long
        # str = sprintf('%s %s %3.10f',t,i,logL);
        # disp(str);
        # disp([num2str(t),' ',num2str(i-1),' ',num2str(logL)])
        # disp([num2str(t),' ',num2str(i-1),' ',sprintf('%3.16f',logL)])
        # std_SSR(t)=sqrt([1,1]*P_T(:,:,t)*[1;1]);
    #

    # log likelihood value to maximize.
    EKF_logL = -0.5 * KT * numpy.log(2*numpy.pi) - 0.5 * logL
    # Negate the log likelihood value because fminunc minimizes.
    EKF_logL = -EKF_logL

    print(EKF_logL*1e-3, Parameters[0:10], Rho12)

    tmp1 = numpy.sum(x_T, axis=0)
    figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
    subplot = figure.add_subplot(1,1,1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)
    subplot.plot(x_T[0,:], linewidth=2, color='blue',  marker='', markersize=3, zorder=1, label="")
    subplot.plot(x_T[1,:], linewidth=2, color='green', marker='', markersize=3, zorder=2, label="")
    subplot.plot(tmp1,     linewidth=2, color='red',   marker='', markersize=3, zorder=3, label="")
    pyplot.savefig("plot.pdf")
    pyplot.show()

    return (EKF_logL,x_T)