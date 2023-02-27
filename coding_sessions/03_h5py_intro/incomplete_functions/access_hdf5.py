import h5py
import os
import numpy as np




def read_dataset( h5_filename, dset_name, folder='/' ):
    """
    Read a dataset and return a copy of it
    the dataset is read from 'h5_filename[ folder dset_name] '
    Parameters:
    -----------
    h5_filename:    string
                    name of the hdf5 file
    dset_name:      string
                    name of the dataset
    folder:         string, default '/'
                    name of the folder where dataset is stored
    Returns:
    --------
    dataset:        numpy nd-array
                    specified dataset

    Example:
    read_dataset( h5_filename, 'image_data/dset_0' )
    read_dataset( h5_filename, 'dset_0', 'image_data/' )
    Both expressions yield the same result
    """
    if folder[-1] != '/':
        folder = folder + '/'
    path_to_file = folder + dset_name
    h5file = h5py.File( h5_filename, 'r' )
    dset = h5file[ path_to_file][:]
    h5file.close()
    return dset



def search_datasets( h5file=None, filename=None):
    """
    Print out every dataset in the specified file.
    Accepts a reference to file as input or the path to the hdf5 file
    If both are given, the function will take the reference to file
    Parameters:
    -----------
    h5file:     h5py.File, default None
                reference to hdf5 file
    filename:   string, default None
                full path to hdf5 file
    Returns:
    --------
    None:       prints out the found datasets
    """
    if filename is None and h5file is None:
        print( 'No input given in "search_datasets". Need at least one input, returning...' )
    if h5file is None:
        h5file = h5py.File( filename, 'r')
    ### Recursively traverse all objects in the hdf5 file
    h5file.visititems( display_datasets)
    if filename is not None:
        h5file.close()



def display_datasets( member_name, h5object):
    """
    Function written for h5py.File.visititems
    Iterates over all objects in the hdf5 file and prints out only datasets
    and which folder they are attached in
    Parameters:
    -----------
    member_name:    string
                    Passed name of the object
    h5obect:        h5py object
                    Corresponding object to the 'member_name' 
                    i.e. h5file[ member_name]
    Returns:
    --------
    None,           prints output to console
    """
    if isinstance( h5object, h5py.Group):
        print( '\nInspecting group:', member_name) 
    if isinstance( h5object, h5py.Dataset): #TODO if the h5object is a Dataset
        dataset = os.path.basename( member_name)

        if os.path.dirname( member_name): #if it is in a subfolder
            print( '\tdataset:', dataset )
        else: #dataset is in root '/' 
            print( '\ndataset in root:', dataset )



    
