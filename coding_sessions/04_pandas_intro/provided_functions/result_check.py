import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt #TMP
from numpy.fft import fft2, ifft2




### pandas dataframe
def nan( df):
    """
    Checks if there are any 'nan' in the dataframe
    also checks if the values have been correctly assigned the median
    by rounding on the second digit after comma
    """
    assert  np.sum( np.sum( df.isnull() ) )==0, 'there are still "NaN" contained in the dataframe, please fill them with the median of the column!'
    df_array = 10*(np.array(df.loc[:, ['grade','semester'] ]) )
    assert (np.ceil( df_array )==np.floor( df_array )).all(), 'replace the "NaN" with the median, not the mean'
    print( 'values correctly filled' )

def df_to_array( A, B):
    """
    Check if to arrays match, if there are any nan in the arrays
    the values will be locally replaced with some dummy values 
    """
    #assert not (np.isnan(A)).any(), 'missing numbers ("nan") occurend in the dataframe, please fix!'
    A = A.round(5)
    A[ np.isnan(A)] = -10
    B = B.round(5)
    B[ np.isnan(B)] = -10
    correct = True
    try:
        correct = np.allclose( A, B)
    except:
        correct = False
    if not correct:
        raise Exception(  'wrong columns have been selected, please correct!' )
    print( 'arrays match' )



def get_gender( df):
    """
    Randomly sample a list/array with genders sampled
    by some probablity
    """
    n_columns = df.shape[0]
    gender = []
    for i in range( n_columns ):
        roll = np.random.rand()
        if roll < 0.6:
            gender.append( 'm')
        elif roll < 0.95:
            gender.append( 'f')
        else:
            gender.append( 'd')
    gender = np.array( gender)
    return gender
    #np.array(gender)


def gender( df):
    """
    check if a gender column has been added to the dataframe
    and if there are legit values in there ( see get_gender for
    reference)
    """
    try:
        gender = np.array( df['gender'] )
    except:
        raise Exception( "'gender' column not found, please add a 'gender' column in the dataframe" )
    if  not ((gender =='m') + (gender =='f') + (gender =='d') ).all() :
        raise Exception( 'wrong values set for the gender, please fix' )
    print( 'gender values correctly filled' )


def attendance( df):
    """
    check if the attendance has been firstly incremented 
    and if it was correctly incremented (capped at 100)
    """
    df_raw = pd.read_csv( 'data/student_info.csv' )
    df_raw = df_raw.fillna( {'attendance %':df_raw['attendance %'].mean() })
    original_attendance = df_raw[ 'attendance %']
    current_attendance = df[ 'attendance %']
    if np.allclose( current_attendance, original_attendance):
        raise Exception( 'Attendance unaltered, please increment it by 10 everywhere')
    elif (current_attendance > 100).any():
        raise Exception( "illegal value found in attendance, attendance can't got above 100%" )
    return


