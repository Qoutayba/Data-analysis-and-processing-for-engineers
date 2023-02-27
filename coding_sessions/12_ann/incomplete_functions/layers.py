import numpy as np
from general_functions import glorot_initializer


class Layer:
    """
    Define the properties and methods of a layer for a dense feed forward
    artificial neural network.
    When the <Layer> class is inhereted, only the activation function and
    its derivative have to be redefined
    """
    __name__ = 'linear' #string for debugging
    def __init__( self, neuron_in, neuron_out, learning_rate=0.05):
        """
        Randomly allocate layer parameters and set other hyperparameters
        Parameters:
        -----------
        neuron_in:      int
                        number of neurons the layer has as input
        neuron_out:     int
                        number of neurons the layer has as output
        learning_rate:  float, default 0.05
                        learning rate of the specified layer
        Returns:
        --------
        None:           only updates self and sets parameters
        """
        #self.weight = #TODO #allocate the weight matrix via glorot initializer
        self.bias = glorot_initializer( neuron_out)[:,None]
        self.learning_rate = learning_rate


    def forward_propagation( self, x, training=False):
        """
        Compute the evaluation of the layer: forward propagation 
        a = f(z) = f( W x + b)
        During training the derivative is stored, which is required 
        for the gradient computation
        Parameters:
        -----------
        x:          numpy 2d-array
                    input sample(s), number of samples has to be written 
                    in the last axis
        training:   bool, default False
                    whether variables required for traning should be stored
        Returns:
        --------
        y:          numpy 2d-array
                    computed output of this layer 
        """
        #a = self.activation_function( #TODO #compute layer output
        if training is True: #required variables for backprop
            self.input = x.copy() 
            self.derivative = self.compute_derivative( a) #store the gradients as attribute
        return #TODO #return the layer output


    def back_propagation( self, delta): 
        """
        Update the delta and adjust the layer parameters in the current
        training step.
        (training must be set to True in <forward_propagation>)
        Parameters:
        -----------
        delta:          numpy 2d-array  
                        delta of the previous layer 'l+1' 
        Returns:
        --------
        delta:          numpy 2d-array
                        delta for the next layer 'l-1' computation of the
                        weight update already conducted
        """
        #Hint: All required terms (partial derivates) and update equations
        #      can be found in the screencast notebook
        #Hint2: Point wise operation and matrix multiplication is required
        #Debugging hint: Write the required shapes of e.g. delta & weights on paper
        #                compare them to the shapes in your code, e.g. print( delta.shape)
        #                Note that each sample has 1 delta term -> delta.shape[-1] == n_samples

        bias_increment   = np.sum( delta, axis=1 )[:,None]
        #weight_increment = #TODO... * self.derivative @ #TODO...
        #delta            = #TODO

        n_samples = delta.shape[-1]
        update_scaling = self.learning_rate / n_samples
        #self.weight   #TODO * update_scaling #update the weights
        #self.bias     #TODO #update the bias
        return delta


    def activation_function( self, z):
        """
        Definition and application of the activation function <f> of the layer
        a = f(z) = z
        Parameters:
        -----------
        z:      numpy 2d-array
                activation of this layer
        Returns:
        --------
        a:      numpy 2d-array
                output of this layer
        """
        return z #linear layer: f(z) = z


    def compute_derivative( self, a):
        """
        Definition and computation of the derivative da/dz
        of the activation function of the layer: f'(z) = 1
        Parameters:
        -----------
        a:      numpy 2d-array
                output of this layer
        Returns:
        --------
        da/dz:  2d-array
                derivative of <a> with respect to <z>
        """
        return np.ones( a.shape) # f(z) = z -> f'(z) = 1



class Sigmoid(Layer):
    __name__ = 'sigmoid'
    def __init__(self, *args, **kwargs):
        """
        allocate the layer as defined by the parent <Layer> object
        to see the input arguments see help( Layer)
        Parameters:
        -----------
        *args, **kwargs:    ensure all parameters are passed to the parent init
        """
        super().__init__( *args, **kwargs)


    def activation_function( self, z):
        """
        Definition and application of the activation function <f> of the layer
        a = f(z) = 1/( 1+ exp( -z) )
        Parameters:
        -----------
        z:      numpy 2d-array
                activation of this layer
        Returns:
        --------
        a:      numpy 2d-array
                output of this layer
        """
        print( 'ATTENTION: Implement the activation (in layers.py) in the subclasses before continuing' )
        print( 'delete these print statements after implemented')
        return #TODO


    def compute_derivative(self, a):
        """
        Definition and computation of the derivative for the activation
        function of the layer: f'(z) = f(z)*(1- f(z) )
        Parameters:
        -----------
        a:      numpy 2d-array
                output of this layer
        Returns:
        --------
        da/dz:  2d-array
                derivative of <a> with respect to <z>
        """
        print( 'ATTENTION: Implement the compute derivative in (in layers.py) the subclasses before continuing' )
        print( 'delete these print statements after implemented')
        return #TODO
 


# Additional documentation see above (otherwise only repetitive)
class Tanh( Layer):
    __name__ = 'tanh'
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)


    def activation_function( self, z):
        #tanh(x) = (exp( 2x) -1) / (exp(2x) +1) 
        print( 'ATTENTION: Implement the activation (in layers.py) in the subclasses before continuing' )
        print( 'delete these print statements after implemented')
        return #TODO


    def compute_derivative( self, a): 
        #f'(x) = 1-f(x)^2
        print( 'ATTENTION: Implement the compute derivative (in layers.py) in the subclasses before continuing' )
        print( 'delete these print statements after implemented')
        return #TODO



class ReLu(Layer):
    __name__ = 'ReLu'
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)


    def activation_function( self, z):
        return np.maximum( 0, z)


    def compute_derivative(self, a):
        da_dz = np.zeros( a.shape)
        da_dz[ a != 0] = 1
        return da_dz


### Definition of additional optional layers
# e.g. leaky relu, selu, ...
# always copy the structure of the layers above
