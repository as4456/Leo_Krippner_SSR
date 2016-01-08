import datetime
import numpy


def AAL_CommonSaveName(DataFileName, ZLB_Imposed, IEKF_Count, SampleMaturities, N, DataFrequency, FINAL, rL):
    #UNTITLED Summary of this function goes here
    #   Detailed explanation goes here

    # Time stamp for file names.
    TimeStamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    if (ZLB_Imposed == 0):
        ModelType = 'ANSM'
    else:
        ModelType = 'KANSM'

    if (FINAL == 1):
        FinalOrInterim = 'Final'
    else:
        FinalOrInterim = 'Interim'

    if (IEKF_Count < 0):
        IEKFString = "_E%.6f" % numpy.log10(-IEKF_Count)
    else:
        IEKFString = "%.6f" % numpy.log10(IEKF_Count)


    if (rL == -10):
        LowerBoundString = 'Est'
    else:
        LowerBoundString = "%.6f" % 10000 * rL


    SaveName = DataFileName + '_rL_' + LowerBoundString + '_%.6f' % SampleMaturities[-1] + '_' + ModelType + "%i" % N + '_' + DataFrequency + '_IEKF' + IEKFString + '_' + FinalOrInterim
    SaveName = SaveName + TimeStamp


    return SaveName



