import numpy as np
import h5py
import os

def memory( dataset):
    """
    check if the variable has been locally assigned or points to the dataset
    """
    if not isinstance( dataset, h5py._hl.dataset.Dataset):
        raise Exception ("Only assign a pointer to the dataset, do not load the whole dataset\n" +
                        'To clear out some hdf5 issues rerun the cell at the top')
def folder( h5file):
    """ check if the results folder has been correctly added """
    if not 'results' in h5file:
        h5file.close()
        raise Exception( 'please create the results group/subfolder' )

def hdf5_file( filename):
    if not os.path.isfile( filename):
        raise Exception ("hdf5-file has not been written, assign the correct permissions to h5py and write the datasets")
    print( '\nfile correctly written\n')


def allocation( dataset):
    """
    check if the given parameter is a dataset and whether the 
    values have been correctly filled (all 1 in this case)
    """
    if not isinstance( dataset, h5py._hl.dataset.Dataset):
        raise Exception( 'reference to dataset has been lost, pointer has to be reassigned' )
    if not np.allclose( dataset, 1):
        raise Exception( 'values have not been correctly set, expected an array of all ones' )
    print( 'pointer to the dataset set')


def metadata( attributes):
    """
    Check if metadata has been added
    assumes that per default one entry is already added and raises a different error
    """
    if len( attributes) == 1:
        raise Exception( 'please add different metadata besides the h5py.version' )
    elif len( attributes) == 0:
        raise Exception( 'no metadata has been found, please add metadata to the object' )
    print( 'metadata nicely set')


def permissions( h5file):
    """
    check if only read permissions have been set to the h5file
    """
    try:
        h5file.create_group( 'debug_test')
        del h5file[ 'debug_test']
        raise Exception( "write permissions in file are accessible, please assign only read 'r' permissions" )
    except:
        return


