import datetime
import numpy


#######################################################################################################################
def datetime_to_matlab_datenum(dt):
    return 366 + 1 + (dt - datetime.datetime(1, 1, 1, 0, 0)).total_seconds() / datetime.timedelta(1).total_seconds()


#######################################################################################################################
def matlab_datenum_to_datetime(dn):
    # Convert a datetime object to its MATLAB datenum() equivalent
    return datetime.datetime(1, 1, 1, 0, 0) + datetime.timedelta(int(dn - 366 - 1))

#######################################################################################################################
