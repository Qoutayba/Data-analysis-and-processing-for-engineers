def hello_world( name='world'):
    print( 'Hello, {}!'.format( name) )
    return


























import numpy as np

def split_data( inputs, outputs, split=0.3, shuffle=True):
    """ 
    Randomly shuffle the data and thereafter split it into two sets 
    Arranges the data row-wise if it is given column wise (return arrays, each row one sample)
    Data is ALWAYS returned row-wise
    (Note that this function assumes that there are more samples than dimensions of the problem)
    Parameters:
    -----------
    inputs:     numpy nd-array
                input data (preferably) arranged row wise
    outputs:    numpy nd-array
                output data (preferably) arranged row wise
    split:      float, default 0.3
                percentage part of the second set (validation set)
    shuffle:    bool, default True
                Whether the data should be randomly shuffled before splitting
    Returns:
    --------
    input_train:    numpy nd-array
                    input data containing 1-split percent of samples (rounded up)
    input_valid:    numpy nd-array
                    input data containing split percent of samples (rounded down)
    output_train:   numpy nd-array
                    output data containing 1-split percent of samples (rounded up)
    output_valid:   numpy nd-array
                    output data containing split percent of samples (rounded down) 
    """
    if inputs.shape[0] < inputs.shape[1]:
        print( 'Transposing inputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged inputs')
        inputs = inputs.T
    if outputs.shape[0] < outputs.shape[1]:
        print( 'Transposing outputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged outputs')
        outputs = outputs.T
    n_data  = inputs.shape[0]
    n_train = ceil( (1-split) * n_data )
    if shuffle is True:
        shuffle = np.random.permutation(n_data)
        x_train = inputs[ shuffle,:][:n_train,:]
        x_valid = inputs[ shuffle,:][n_train:,:]
        y_train = outputs[ shuffle,:][:n_train,:]
        y_valid = outputs[ shuffle,:][n_train:,:]
        return x_train, y_train, x_valid, y_valid
    else:
        return inputs[:n_train,:], inputs[n_train:,:], outputs[:n_train,:], outputs[n_train:,:]

