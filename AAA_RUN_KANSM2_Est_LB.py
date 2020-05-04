from globl import *
from functions import *
from AAB_KAGM_Estimation_NelderMead_file import AAB_KAGM_Estimation_NelderMead
from AAC_KAGM_SingleLoop_file import *
from AAH_EMS_N23_function_file import *
from AAL_CommonSaveName_file import *
from AAF_FiniteDifferenceHessian_file import *

import numpy
import datetime
import math
import scipy
import scipy.optimize
import csv

# Hyperparameters.
N = 2  # FIXED: Number of factors.
dTau = 0.01  # OPTION: Spacing for TauGrid, used to numerically obtain R.
KappaP_Constraint = 'Direct'  # FIXED: KappaP matrix values are set directly, (but subject to an eigenvalue constraint in 'AAC_EKF_CAB_GATSM_SingleLoop').
ZLB_Imposed = 1  # FIXED: 0=ANSM(2) or 1=K-ANSM(2).
DailyIterations = 200  # OPTION: Sets number of iterations between interim saves.
IEKF_Count = -1e-5  # OPTION: EKF if 0, IEKF steps if >0, tolerance if <0 (e.g. -1e-5).
FinalNaturalParametersGiven = 1  # OPTION: Full estimation if 0 (the Optimization toolbox is required), partial estimation with given parameters if 0.
HessianRequired = 0  # OPTION: Omits Hessian and standard errors if 0, calculates them if 1.

# GSW US data file.
Country = 'UK'
DataFrequency = 'Monthly'

DataFileName = Country + '_GSW_Govt'

# load([DataFileName,'.mat'])
# DailyDateIndex DailyYieldCurveData Maturities MonthlyDateIndex MonthlyYieldCurveData WeeklyDateIndex WeeklyYieldCurveData
Maturities = numpy.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30])
sub_datenum = datetime.date(1899, 12, 30).toordinal() + 366
daily_file_name = Country + '_Daily.csv'
daily_data = numpy.genfromtxt(daily_file_name, delimiter=',')

DailyDateIndex = daily_data[:, 0].copy().astype('int')
DailyDateIndex = DailyDateIndex.__add__(sub_datenum)
DailyYieldCurveData = daily_data[:, 1:].copy()

weekly_file_name = Country + '_Weekly.csv'
weekly_data = numpy.genfromtxt(weekly_file_name, delimiter=',')

WeeklyDateIndex = weekly_data[:, 0].copy().astype('int')
WeeklyDateIndex = WeeklyDateIndex.__add__(sub_datenum)
WeeklyYieldCurveData = weekly_data[:, 1:].copy()

month_file_name = Country + '_Monthly.csv'
month_data = numpy.genfromtxt(month_file_name, delimiter=',')

MonthlyDateIndex = month_data[:, 0].copy().astype('int')
MonthlyDateIndex = MonthlyDateIndex.__add__(sub_datenum)
MonthlyYieldCurveData = month_data[:, 1:].copy()

# print(numpy.shape(DailyDateIndex))
# print(numpy.shape(DailyYieldCurveData))
# print(numpy.shape(Maturities))
# print(numpy.shape(MonthlyDateIndex))
# print(numpy.shape(MonthlyYieldCurveData))
# print(numpy.shape(WeeklyDateIndex))
# print(numpy.shape(WeeklyYieldCurveData))


FirstDay = DailyDateIndex[0]
LastDay = DailyDateIndex[-1]
# FirstDay = datetime_to_matlab_datenum(datetime.datetime.strptime('30-Dec-1994', "%d-%b-%Y")) # Start of 30-year data.
# LastDay  = datetime_to_matlab_datenum(datetime.datetime.strptime('31-Jul-2013', "%d-%b-%Y"))
SampleMaturities = numpy.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 30])

# Set starting parameters.
# These lines set the estimated values from Krippner (2015).

# LoadName='BOOK_US_GSW_Govt_OIS_rL_30_K_AFNSM2_Monthly_IEKF_E-5_Final_2014_8_31_10_52_X';
# load([LoadName,'.mat'],'FinalNaturalParameters');
# FinalNaturalParameters
FinalNaturalParameters_country = "FinalNaturalParameters_" + Country + ".dat"
FinalNaturalParameters = numpy.loadtxt(FinalNaturalParameters_country)
InitialNaturalParameters = numpy.copy(FinalNaturalParameters)

# Select the required data from the yield curve data file.
IncludeMaturities = numpy.array([x in SampleMaturities for x in Maturities])
if DataFrequency == 'Daily':
    (StartT,) = numpy.where(DailyDateIndex == FirstDay)
    (EndT,) = numpy.where(DailyDateIndex == LastDay)
    YieldCurveDateIndex = DailyDateIndex[StartT:EndT + 1]
    YieldCurveData = DailyYieldCurveData[StartT:EndT + 1, IncludeMaturities]
    Dt = (YieldCurveDateIndex[-1] - YieldCurveDateIndex[0] + 1) / (len(YieldCurveDateIndex) * 365.25)
    Iterations = numpy.copy(DailyIterations)
elif DataFrequency == 'Weekly':
    # Find earliest Friday consistent with FirstDay.
    ReferenceFriday = datetime_to_matlab_datenum(datetime.datetime.strptime('24-May-2013', "%d-%b-%Y"))
    WeeksToStepBackForStart = numpy.floor((ReferenceFriday - FirstDay) / 7)
    FirstWeek = ReferenceFriday - 7 * WeeksToStepBackForStart
    # Find latest Friday consistent with LastDay.
    WeeksToStepBackForEnd = 1 + numpy.floor((ReferenceFriday - LastDay) / 7)
    LastWeek = ReferenceFriday - 7 * WeeksToStepBackForEnd
    (StartT,) = numpy.where(WeeklyDateIndex == FirstWeek)
    (EndT,) = numpy.where(WeeklyDateIndex == LastWeek)
    YieldCurveDateIndex = WeeklyDateIndex[StartT:EndT + 1]
    YieldCurveData = WeeklyYieldCurveData[StartT:EndT + 1, IncludeMaturities]
    Dt = 7 / 365.25
    Iterations = DailyIterations * 5
else:
    # Find earliest end-month consistent with FirstDay.
    [FirstYear, FirstMonth] = [datetime.datetime.fromordinal(numpy.int(FirstDay - 366)).year,
                               datetime.datetime.fromordinal(numpy.int(FirstDay - 366)).month]
    if FirstMonth == 12:
        FirstYear = FirstYear + 1
        FirstMonth = 0
    FirstMonth = numpy.int(datetime_to_matlab_datenum(datetime.datetime(FirstYear, FirstMonth + 1, 1)) - 1)

    # Find latest end-month consistent with LastDay.
    [LastYear, LastMonth] = [datetime.datetime.fromordinal(numpy.int(LastDay - 366)).year,
                             datetime.datetime.fromordinal(numpy.int(LastDay - 366)).month]
    if LastMonth == 12:
        LastYear = LastYear + 1
        LastMonth = 0
    LastMonth1 = numpy.int(datetime_to_matlab_datenum(datetime.datetime(LastYear, LastMonth + 1, 1)) - 1)
    if LastMonth1 != LastDay:
        if LastMonth == 0:
            LastYear = LastYear - 1
            LastMonth = 12
        LastMonth1 = datetime_to_matlab_datenum(datetime.datetime(LastYear, LastMonth, 1)) - 1
    (StartT,) = numpy.where(MonthlyDateIndex == FirstMonth)
    (EndT,) = numpy.where(MonthlyDateIndex == LastMonth1)
    if (numpy.size(StartT) > 0) and (numpy.size(EndT) > 0):
        YieldCurveDateIndex = MonthlyDateIndex[StartT[0]:EndT[0] + 1]
        YieldCurveData = MonthlyYieldCurveData[StartT[0]:EndT[0] + 1, IncludeMaturities]
    else:
        YieldCurveDateIndex = numpy.array([])
        YieldCurveData = numpy.array([])
    Dt = 1.0 / 12
    Iterations = DailyIterations * 21

Tau_K = numpy.copy(SampleMaturities)

# Estimation.
if FinalNaturalParametersGiven == 1:
    print('Finalizing model K-AFNSM(2) for ' + Country + " using %i" % Maturities[0] + "-%i" % SampleMaturities[
        -1] + ' year data at ' + DataFrequency + ' frequency for period ' + matlab_datenum_to_datetime(
        YieldCurveDateIndex[0]).strftime('%d-%b-%Y') + ' to ' + matlab_datenum_to_datetime(
        YieldCurveDateIndex[-1]).strftime('%d-%b-%Y'))
    FinalNaturalParameters = numpy.copy(InitialNaturalParameters)
    FINAL = 1
    Exitflag = -1
    [Fval, x_T] = AAC_KAGM_SingleLoop(numpy.matrix(YieldCurveData), Tau_K, N, FinalNaturalParameters, Dt, dTau,
                                      KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)

    Time0 = 0
    Time1 = 0
    Output = 'Final parameters given'
    rL = FinalNaturalParameters[0]
    KappaQ2 = FinalNaturalParameters[1]
    KappaP = numpy.array([[FinalNaturalParameters[2], FinalNaturalParameters[3]],
                          [FinalNaturalParameters[4], FinalNaturalParameters[5]]])
    ThetaP = numpy.array([[FinalNaturalParameters[6]], [FinalNaturalParameters[7]]])
    Sigma1 = FinalNaturalParameters[8]
    Sigma2 = FinalNaturalParameters[9]
    Rho12 = FinalNaturalParameters[10]
else:
    # Estimate final parameters.
    print('Estimating K-AFNSM(2) for ' + Country + " using %i" % SampleMaturities[0] + "-%i" % SampleMaturities[
        -1] + ' year data at ' + DataFrequency + ' frequency for period ' + matlab_datenum_to_datetime(
        YieldCurveDateIndex[0]).strftime('%d-%b-%Y') + ' to ' + matlab_datenum_to_datetime(
        YieldCurveDateIndex[-1]).strftime('%d-%b-%Y'))
    Exitflag = 0
    while (Exitflag == 0):
        if (KappaP_Constraint == 'Direct'):
            InitialParameters = numpy.copy(InitialNaturalParameters)
            tmp_func = lambda x: x / (1 + numpy.abs(x)) - InitialNaturalParameters[9]
            InitialParameters[9] = scipy.optimize.fsolve(tmp_func, 1)
        elif (KappaP_Constraint == 'S/A'):
            print('Nothing here.')

        # Extended Kalman filter estimation.
        # Time0 = matlab_now(datetime.datetime.now())
        Time0 = datetime_to_matlab_datenum(datetime.datetime.now())
        FINAL = 0
        Max_IEKF_Count = 0
        Max_IEKF_Point = 0

        [x_T, FinalParameters, Fval, Exitflag, Output] = AAB_KAGM_Estimation_NelderMead(YieldCurveData, Tau_K, N,
                                                                                        InitialParameters, Dt, dTau,
                                                                                        KappaP_Constraint, ZLB_Imposed,
                                                                                        IEKF_Count, FINAL, Iterations)
        Time1 = matlab_now(datetime.datetime.now())

        if (KappaP_Constraint == 'Direct'):
            # Take the absolute value of Sigma parameters.
            FinalNaturalParameters = FinalParameters;
            FinalNaturalParameters[8] = abs(FinalParameters[8])
            FinalNaturalParameters[9] = abs(FinalParameters[9])
            # Convert correlation parameters into correlations.
            FinalNaturalParameters[10] = FinalParameters[10] / (1 + abs(FinalParameters[10]))
            FinalNaturalParameters[11:] = abs(FinalParameters[11:])
            # Calculate the state equation quantities based on parameter values.
            KappaQ = numpy.array([[0, 0], [FinalNaturalParameters[0], 0]])
            KappaP = numpy.array([[FinalNaturalParameters[1], FinalNaturalParameters[2]],
                                  [FinalNaturalParameters[3], FinalNaturalParameters[4]]])
            D, V = numpy.linalg.eig(KappaP)
            d1 = D[0, 0]
            d2 = D[1, 1]
            if ((d1.real < 0) or (d2.real < 0)):
                if (d1.real < 0):
                    d1 = 1e-6 + d1.imag * 1j
                if (d2.real < 0):
                    d2 = 1e-6 + d2.imag * 1j

                D = numpy.diag([d1, d2])
                tmp1 = numpy.matrix(V)
                tmp2 = tmp1 * numpy.matrix(numpy.reshape(D, (len(D), 1)))
                KappaP = numpy.array(numpy.real(numpy.linalg.solve(tmp1.getH(), tmp2.getH()).getH()))
                FinalNaturalParameters[2] = KappaP[0, 0]
                FinalNaturalParameters[3] = KappaP[0, 1]
                FinalNaturalParameters[4] = KappaP[1, 0]
                FinalNaturalParameters[5] = KappaP[1, 1]

        elif (KappaP_Constraint == 'S/A'):
            print('Nothing here')

    rL = FinalNaturalParameters[0]
    KappaQ2 = FinalNaturalParameters[1]
    KappaP = numpy.array([[FinalNaturalParameters[2], FinalNaturalParameters[3]],
                          [FinalNaturalParameters[4], FinalNaturalParameters[5]]])
    ThetaP = numpy.array([[FinalNaturalParameters[6]], [FinalNaturalParameters[7]]])
    Sigma1 = FinalNaturalParameters[8]
    Sigma2 = FinalNaturalParameters[9]
    Rho12 = FinalNaturalParameters[10]

    # disp(Exitflag)
    print([Max_IEKF_Point, Max_IEKF_Count])
    print(Fval)
    print(FinalNaturalParameters[0:10])

    # plotyy(1:length(x_T),x_T',1:length(x_T),sum(x_T)')
    # pause 0.1
    # Create figure
    figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')

    subplot1 = figure.add_subplot(1, 1, 1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)
    tmp1 = numpy.arange(0, max(numpy.shape(x_T)) + 1)
    subplot1.plot(tmp1, x_T.getH(), linewidth=2, marker='', markersize=3, zorder=1, label="")
    subplot2 = subplot1.twinx()
    subplot1.plot(tmp1, numpy.sum(x_T).getH(), linewidth=2, marker='', markersize=3, zorder=1, label="")
    SaveName = AAL_CommonSaveName(DataFileName, ZLB_Imposed, IEKF_Count, SampleMaturities, N, DataFrequency, FINAL,
                                  -10);
    disp(SaveName)

    figure1 = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
    figure1.plot(tmp1, x_T.getH(), linewidth=2, marker='', markersize=3, zorder=1, label="")
    # Save final output in CSV file.
    NaturalParameterStandardErrors = -9.999 * numpy.ones(numpy.size(FinalNaturalParameters))
    numpy.savetxt(SaveName + ".csv", FinalNaturalParameters, fmt='%25.15e', delimiter=',', newline='\n')

    # Reset InitialNaturalParameters for next iteration.
    InitialNaturalParameters = numpy.copy(FinalNaturalParameters)
#

# Diagnostics and output.
# Calculate Hessian and standard errors for parameters, if required.
if (HessianRequired == 1):
    # Calculate Hessian and standard errors for natural model parameters.
    FINAL = 1
    NaturalHessian = AAF_FiniteDifferenceHessian(AAC_KAGM_SingleLoop, FinalNaturalParameters, 1e-10, YieldCurveData,
                                                 Tau_K, N, Dt, dTau, KappaP_Constraint, ZLB_Imposed, IEKF_Count, FINAL)
    NaturalParameterStandardErrors = numpy.sqrt(numpy.abs(numpy.diag(numpy.linalg.inv(NaturalHessian))));
else:
    NaturalParameterStandardErrors = -9.999 * numpy.ones(numpy.size(FinalNaturalParameters))

# Display output.
dTime = Time1 - Time0
print(dTime * 24, 'hours (=', dTime * 24 * 60, 'minutes)')
print(Output)
print(Exitflag)
print(Fval)
print(InitialNaturalParameters[0:10])
print(FinalNaturalParameters[0:10])
print(NaturalParameterStandardErrors[0:10])
KappaQ = numpy.matrix([[0, 0], [0, KappaQ2]])
print(KappaP, numpy.linalg.eig(KappaP), KappaQ - KappaP)

(T, K) = numpy.shape(YieldCurveData)
Residuals = numpy.ones((T, K)) * float('nan')
PlotCurves = 0

for t in range(0, T):
    YieldCurveData_t = YieldCurveData[t, :]
    (Fitted_R_t, tmp1) = AAD_KAGM_R_and_dR_dx(x_T[:, t], rL, KappaQ2, Sigma1, Sigma2, Rho12, Tau_K, dTau, ZLB_Imposed)
    Fitted_R_t = numpy.reshape(numpy.array(Fitted_R_t), (numpy.size(Fitted_R_t),))

    if (PlotCurves == 1):
        figure = pyplot.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w')
        subplot = figure.add_subplot(1, 1, 1, position=[0.15, 0.10, 0.75, 0.80], frame_on=True, zorder=0)
        subplot.plot(Tau_K, 100 * Fitted_R_t, linewidth=2, marker='', markersize=3, zorder=1, label="")
        subplot.plot(Tau_K, YieldCurveData_t, linestyle='', linewidth=2, marker='o', markersize=3, zorder=1, label="")
        # ylim([-2 10]);
        # pause(0.3)

    Residual_t = 0.01 * YieldCurveData_t - Fitted_R_t
    print(numpy.shape(Residuals))
    print(numpy.shape(Residual_t))
    Residuals[t, :] = Residual_t
#

RMSE_Residuals = numpy.sqrt(numpy.sum(numpy.multiply(Residuals, Residuals)) / T)
print(numpy.mean(Residuals))
print(RMSE_Residuals)

Phi = KappaQ2
(SSR, EMS_Q, ETZ_Q) = AAH_EMS_N23_function(Phi, x_T, dTau)

# Save final output in MatLab file.
SaveName = AAL_CommonSaveName(DataFileName, ZLB_Imposed, IEKF_Count, SampleMaturities, N, DataFrequency, FINAL, -10);
print(SaveName)

# Save final output in CSV file.
# numpy.savetxt(SaveName+".csv", FinalNaturalParameters, fmt='%25.15e', delimiter=',', newline='\n')
# save(SaveName)

# Save final output in Excel spreadsheet.
RangeName = 'A2:O%i' % (max(numpy.shape(YieldCurveDateIndex)) + 1)
# xlswrite([SaveName,'.xlsm'],[YieldCurveDateIndex-datenum('30-Dec-1899'),YieldCurveData,100*x_T',100*SSR,100*EMS_Q,ETZ_Q],'D. Monthly',RangeName)

SSR_pos = SSR.copy()
SSR_pos[SSR_pos <= 0] = numpy.nan
fig, ax = pyplot.subplots()
pyplot.plot(SSR_pos * 100, linewidth=2, marker='', markersize=3, zorder=1, label="slope", color='b')
SSR_neg = SSR.copy()
SSR_neg[SSR_neg > 0] = numpy.nan
pyplot.plot(SSR_neg * 100, linewidth=2, marker='', markersize=3, zorder=1, label="slope", color='r')

a = ax.get_xticks()
y = [matlab_datenum_to_datetime(YieldCurveDateIndex[int(x)]).date().year for x in a[:-1]]
pyplot.xticks(a, y, rotation=90)
pyplot.ylabel('Percentage')
fig.suptitle('SSR', fontsize=20)
fig.savefig('SSR_' + DataFrequency + '.jpg')

fig, ax = pyplot.subplots()
pyplot.plot(EMS_Q * 100, linewidth=2, marker='', markersize=3, zorder=1, label="slope")

a = ax.get_xticks()
y = [matlab_datenum_to_datetime(YieldCurveDateIndex[int(x)]).date().year for x in a[:-1]]
pyplot.xticks(a, y, rotation=90)
pyplot.ylabel('Percentage')
fig.suptitle('EMS', fontsize=20)
fig.savefig('EMS_' + DataFrequency + '.jpg')

fig, ax = pyplot.subplots()
pyplot.plot(ETZ_Q, linewidth=2, marker='', markersize=3, zorder=1, label="slope")

a = ax.get_xticks()
y = [matlab_datenum_to_datetime(YieldCurveDateIndex[int(x)]).date().year for x in a[:-1]]
pyplot.xticks(a, y, rotation=90)
pyplot.ylabel('Years')
fig.suptitle('ETZ', fontsize=20)
fig.savefig('ETZ_' + DataFrequency + '.jpg')

pyplot.show()

output = numpy.concatenate((YieldCurveData, 100 * x_T.T), axis=1)
output = numpy.concatenate((output, 100 * SSR.reshape((SSR.shape[0], 1))), axis=1)
output = numpy.concatenate((output, 100 * EMS_Q.reshape((SSR.shape[0], 1))), axis=1)
output = numpy.concatenate((output, 100 * ETZ_Q.reshape((SSR.shape[0], 1))), axis=1)

date_str = [matlab_datenum_to_datetime(x).date().__str__() for x in YieldCurveDateIndex]
k = numpy.array(date_str)
k = k.reshape((len(k), 1))
with open(SaveName + '_final.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    header = ['Date'] + [str(x) for x in SampleMaturities] + ['Level', 'Slope', 'SSR', 'EMS-Q', 'ETZ-Q']
    spamwriter.writerow(header)
    for i in range(len(date_str)):
        data_to_write = list(k[i]) + list(output[i])
        spamwriter.writerow(data_to_write)

print('Finished')
