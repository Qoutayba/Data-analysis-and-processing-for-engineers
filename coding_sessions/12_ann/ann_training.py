import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.extend( ['provided_functions', 'incomplete_functions' ] )
sys.path.append( '../submodules' )

import data_processing as process
from layers import *
from general_functions import tic, toc

#NOTE: before filling this code, complete the pre_processing.ipynb notebook

### Data preprocessing
# from the remaining training data, split the validation set
data             = list( np.load( 'data/training_data.npz' ).values() ) #[input, output]
validation_split = #TODO #choose a good parameter
x_train, y_train, x_valid, y_valid = process.split_data( *data, validation_split ) 

# compute a useful split on the training data, apply it to the validation data
input_scaletype  = None #TODO
output_scaletype = None #TODO
x_train, x_valid, input_shift = process.scale_data( #TODO ) 
y_train, y_valid, output_shift = process.scale_data( #TODO )

# do not forget to save the computed shifts
# shift lists will be stored in a dictionary
shift_file = open( 'data/shifts.pkl', 'wb' ) #python objects (lists) are best stored with pickle
pickle.dump( dict( input_shift=input_shift, output_shift=output_shift), shift_file )
shift_file.close()


### List of TODO's:
### It is advised to go through the TODOs step by step.  
"""
## Coding tasks in incomplete_functions/layers.py
- implement weight allocation in Layer.__init__ using the glorot initializer
- implement the activation function and their derivatives in the 'Sigmoid' and 'Tanh' layer classes
- implement forward and backpropagation in the Layer class (will be inhereted to all other layers)
## Coding tasks in this file (ann_training.py)
- implement the component wise derivative of the MSE
- implement the training below in the 'training for loop'
- adjust the hyperparameters to achieve a good model
- save the model/ model parameters after training

## Optional improvements of the ANN
- implement a dynamic learning rate (e.g. start with high learning rate, decay linearly to a constant)
- implement early stopping (e.g. terminate training after n epochs of no improvement)
- implement different layers (e.g. selu, leaky relu, ... )
- implement regularization (e.g. regularized loss, dropout, ...)

## Quality reference:
- a MRE of ~2% in the first two components of the output is a good model
- Note that the loss is computed on the shifted data -> no real world interpretability
"""



### Hyperparameter tuning/ANN definition
#TODO tune the hyperparameters
n_batches          = 1 
data_stochasticity = 0.0 #% of samples chosen in each epoch (stochastic gradient descent)
learning_rate      = 0.005
n_epochs           = 100
hidden_neurons     = [ 5]
connecting_layers  = [ Sigmoid]
connecting_layers += [ Layer] #always linear connection to output
loss               = lambda y_pred, y_true: 1/2 * ( y_pred - y_true )**2 
loss_derivative    = lambda y_pred, y_true: #TODO #component wise derivative of the loss dPhi/da^L 

n_layers     = len( connecting_layers) 
n_neurons    = [x_train.shape[0], *hidden_neurons, y_train.shape[0] ]
architecture = []
for i in range( n_layers):
    architecture.append( connecting_layers[i]( n_neurons[i], n_neurons[i+1], learning_rate=learning_rate) )

## Additional hyperparameters you could implement
#learning_rate_decay =...
#dropout = 0.5 
#regularized_loss = lambda ...


### Training preallocations
debug_timer     = 250
training_loss   = []
validation_loss = []
best_loss       = 1e5
best_epoch      = 0
best_weights    = n_layers*[None]
best_bias       = n_layers*[None]
tic( 'training', silent=False )
tic( '{} additional epochs'.format( debug_timer), silent=True)




for i in range( n_epochs):
    ### Training
    batch_loss = []
    for x_batch, y_batch in process.batch_data( x_train, y_train, n_batches, stochastic=data_stochasticity):
        #forward propagation
        y = x_batch.copy()
        #for layer in architecture: #FORWARD propagation
            #y = layer.forward_propagation( #TODO, training=True)  #implement the forward propagation
        #back propagation
        delta = loss_derivative( y, y_batch) 
        #for #TODO #gradient BACKWARD propagation algorithm
            #delta = #TODO
        batch_loss.append( np.mean( loss( y, y_batch ) ) )

    ### Epoch evaluation
    training_loss.append( sum( batch_loss)/n_batches )
    y = x_valid.copy()
    #for #TODO... #evaluation of (x_valid)
    #validation_loss.append( np.mean( loss( #TODO) ) )
    if None: #TODO < best_loss: #store the model if the validation loss has improved
        best_loss  = validation_loss[-1]
        best_epoch = i
        for j in range( n_layers): #store parameters of the best model
            best_weights[j] = architecture[j].weight.copy()
            best_bias[j]    = architecture[j].bias.copy()

    ### console output
    if (i+1) % debug_timer == 0:
        print( 'trained for a total of {} epochs'.format( i+1) )
        toc( '{} additional epochs'.format( debug_timer), precision=3 )
        tic( '{} additional epochs'.format( debug_timer), silent=True)
        print( 'current training loss:   {:.7f}'.format( training_loss[-1] ))
        print( 'current validation loss: {:.7f}'.format( training_loss[-1] ))
        if (i+1) % 1000 ==0: print()

    #if no improvement in a while: break #optional early stop #TODO




### Training finished, console output and losses
trained_epochs = len( training_loss)
#restore the previously best model and save it
for i in range( n_layers):
    architecture[i].weight = best_weights[i]
    architecture[i].bias   = best_bias[i]
#store the best model (either the layers list with pickle, or only the parameters with numpy) 
#TODO #store the model

## After finding the final ANN, fill out the 'post_processing.py file

print( '\n--------------------------------------------------------')
print( 'Training of {} epochs completed, printing out some values:'.format( trained_epochs) )
toc( 'training' )
print( 'best epoch: {}'.format( best_epoch+1) )
print( 'validation loss: {:.6f}'.format( validation_loss[ best_epoch]) )
print( 'training loss:   {:.6f}'.format( training_loss[ best_epoch]) )
print( 'number of hidden layers: {}'.format( len( architecture) -1 ) )
print( 'hidden neurons: {}'.format( hidden_neurons) )
try: print( 'activation functions: {}'.format( [layer.__name__ for layer in architecture] ) )
except: pass 
print( 'total number of hidden neurons: {}'.format( sum( hidden_neurons) ) )
print( '--------------------------------------------------------')
        
fig, ax = plt.subplots( 1, 1)
ax.semilogy( np.arange( trained_epochs) +1, training_loss, label='training loss')
ax.semilogy( np.arange( trained_epochs) +1, validation_loss, label='validation loss ') 
ax.set_ylim( ymin=min( training_loss) )
ax.grid()
ax.legend()
ax.set_xlabel( '# epochs trained [-]' )
ax.set_ylabel( 'loss [-]' )
plt.show() 
np.savez_compressed( 'data/ann_training_losses.npz', training_loss, validation_loss )
