import h5py
import os

def display_file_contents(filename):
        h5file = h5py.File( filename, 'r')
        display_all_data(h5file)
        h5file.close()

def display_all_data(h5file):
    """ 
    Combines all display-functions to display every stored dataset and metadata
    Input: h5file - opened file object of h5py
    Output: Terminal Output, every data in the file
    """
    print('\n#### Datasets and metadata in root folder ####' )
    display_root(h5file) #displays only datasets and 
    print('\n#### Datasets stored in each folder ####' )
    h5file.visititems(datasets)
    # display only the metadata/attributes
    print('\n#### Metadata stored in each folder ####' )
    h5file.visititems(attrs)
    
def attrs(name, file_obj):
    """
    Function specifically created for the h5py.File.visititems(func) command
    USAGE: call with h5py.File.visititems(print_attrs)
    Output: Terminal output, every metadata and which file it is attached to
    """
    if file_obj.attrs.keys():
        print( '\nMetadata attached to:', name)
        for key in file_obj.attrs.keys():
            print("    %s: %s" % (key, file_obj.attrs[key]) )


def datasets(name, file_obj):
    """
    Function specifically created for the h5py.File.visititems(func) command
    USAGE: call with h5py.File.visititems(print_datasets)
    Output: Terminal output, every dataset and in which folder it is located
    """
    try:
        dirs = list(file_obj.keys())
        i = 0
        found_ds = False
        while i < len(dirs) and not found_ds:
            found_ds = isinstance(file_obj[dirs[i]], h5py.Dataset)
            i += 1

        if found_ds:
            print( '\ncontent of folder: %s' % name)
            [print(file_obj[datasets]) for datasets in dirs if isinstance(file_obj[datasets], h5py.Dataset)]
    except:
        pass


def display_root(h5file):
    """
    Function for hdf5 files, used to display all datafiles and metadata in the root folder
    Does ignore all folders on output!

    Combines all display-functions to display every stored dataset and metadata
    Input: h5file - opened file object of h5py
    Output: Terminal Output, every data in the root folder
    """
    try:
        dirs = list( h5file.keys() )
        i = 0
        found_ds = False
        while i < len(dirs) and not found_ds:
            found_ds = isinstance(h5file[dirs[i]], h5py.Dataset)
            i += 1

        if found_ds:
            print('\nDatasets of the root directory:')
            [ print(h5file[x]) for x in dirs if isinstance(h5file[x], h5py.Dataset)]
        if h5file.attrs.keys():
            print('\nAttributes/Metadata in the root directory:')
            for metakeys in h5file.attrs.keys():
                print('    {}: {}'.format( metakeys, h5file.attrs[metakeys]) )
    except: 
        pass

