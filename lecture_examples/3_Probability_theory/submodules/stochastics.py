import numpy as np

from numpy import pi, exp, sqrt
from math import ceil, floor
from scipy.special import gamma
from scipy.stats import t as t_distribution
from scipy.stats import norm as normal_distribution

    

class estimators: 
    def point_normal_estimation( x):
        """
        Estimate the Parameters of the normal distribution based off the
        maximum likelihood estimator (MLE)
        Parameters:
        -----------
        x:      numpy nd-array
                If x is not a 1d-array, samples have to be arranged row wise
                (each row one sample)
        Returns:
        --------
        expectation:    float or numpy 1d-array
                        estimated expectation of the normal 
                        distribution based off the samples
        variance:       float or numpy 1d-array
                        estimated variance of the normal 
                        distribution based off the samples
        """
        n_samples = x.shape[0]
        return x.mean(0), x.std(0)
        

    def interval_expectation_estimation( x, confidence_level=0.95, theory=None):
        """
        Estimate the Parameters of the normal distribution based off the
        maximum likelihood estimator (MLE)
        Parameters:
        -----------
        x:      numpy nd-array
                If x is not a 1d-array, samples have to be arranged row wise
                (each row one sample)
        confidence_level:   float, default 0.95
                            confidence level of the estimated interval
        theory:             string, default None
                            based on which theory the interval should be estimated
                            choose between:
                            - MLE/maximum_likelihood estimator
                            - t-distribution
                            If no value is given it is chosen by the number of samples 
        Returns:
        --------
        expectation_interval:   list of floats
                                Interval of the estimated expectation
        """
        n_samples = x.shape[0]
        if theory is None:
            if n_samples > 30:
                theory = 'MLE'
            else:
                theory = 't-distribution'

        q = (1+ confidence_level)/2
        expectation, variance = MLE_normal_distribution( x)
        if theory.lower() == 'mle' or 'maximum likelihood' in theory.lower():
            delta = normal_distribution.ppf( q) * variance/n_samples**0.5
        else:
            nu = n_samples -1
            delta = t_distribution( nu).ppf( q) * variance/n_samples**0.5 
        return [ expectation-delta, expectation+delta ]
            


class sample_distribution:
    def normal_distribution( n, mu=0, variance=1):
        """
        samples n values out of the normal distribution
        with the given parameters
        Parameters:
        -----------
        n:          int
                    number of samples
        mu:         float, default 0
                    mu parameter (mean)
        variance:   float, default 1
                    variance parameter (sigma^2)
        Returns:
        --------
        x:          numpy 1d-array
                    discrete samples of the normal distribution
        """
        sigma = variance**0.5
        x = sigma* np.random.randn( n) + mu
        return x


    def uniform_distribution( n, a=0, b=1):
        """
        sample n values out of the uniform distribution
        shadows the numpy function
        Parameters:
        -----------
        a:          float, default 0
                    lower bound of the distribution
        b:          float, default 1
                    upper bound of the distribution
                    expected value of the normal distribution
        interval:   list of 2 floats, default None
                    interval on which to plot the distribution 
        Returns:
        --------
        x:          numpy 1d-array
                    samples of the uniform distribution
        """
        x = np.random.uniform( a, b, n) 
        return x


    def lognormal_distribution( n, mu=0, sigma=1):
        """
        samples n values out of the normal distribution
        shadows the numpy function
        Parameters:
        -----------
        n:          int
                    number of samples
        mu:         float, default 0
                    mu parameter 
        sigma:      float, default 1
                    sigma parameter 
        Returns:
        --------
        x:          numpy 1d-array
                    discrete samples of the normal distribution
        """
        return np.random.lognormal( mu, sigma, size=n)



class distribution_plots: 
    def underlying(x, bins=200):
        """
        Redefine the samples into discrete bins to be able
        to plot the distribution given with discrete samples
        Parameters:
        -----------
        x:      numpy 1d-array
                given samples of the distribution
        bins:   int, default 100
                number of discretized bins to interpolate the plot with
        Returns:
        --------
        x:      numpy 1d-array
                interval spanned by x (input)
        y:      numpy 1d-array        
                underlying distribution function of x
        """
        x = np.squeeze(x) #x is required to be of shape (n,)
        x = np.sort(x)
        n = len(x)
        bins = min(n/3, bins) #ensures that there are not too rough jumps
        f = np.zeros( (bins) )
        delta_x = (x[-1]-(x[0]) )/bins
        scalefactor = ((x[-1]-x[0])/(bins)) *n
        for i in range(bins):
            x_bin = x[0]+i*delta_x 
            f[i] = (np.searchsorted(x, max(x_bin+delta_x,x[0]) ) - np.searchsorted(x, min(x_bin,x[-1]) ) )/scalefactor

        x_dist = np.arange(x[0], x[-1], (x[-1]-x[0])/(bins))[:bins]
        x_dist = x_dist + (x_dist[1]- x_dist[0])/2 #moves each value in the center of the 'bin', not the beginning

        return x_dist, f


    def normal( mu=0, variance=1, interval=None):
        """
        Plot the normal distribution based on its parameters
        on the given interval
        Parameters:
        -----------
        mu:         float, default 0
                    expected value of the normal distribution variance:   float, default 0
                    variance of the normal distribution (sigma^2)
        interval:   list three floats, default None
                    interval of sample defined as [start,stop,step]
        Returns:
        --------
        x:          numpy 1d-array
                    interval spanned by x (input)
        y:          numpy 1d-array        
                    underlying distribution function of x
        """
        sigma = variance**0.5
        if interval is not None:
            x = np.arange( *interval )
        else: 
            x = np.arange( min(-5*sigma, -0.5), max(5*sigma,0.5), (10*sigma)/100 ) +mu 
        phi = 1/ (np.sqrt( 2*pi * variance)) * np.exp( -(x-mu)**2 / (2*variance) ) 
        return x, phi


    def uniform( a=0, b=1, interval=None):
        """
        sample arrays to plot the normal distribution 
        based on its parameters on the given interval
        Parameters:
        -----------
        a:          float, default 0
                    lower bound of the distribution
        b:          float, default 1
                    upper bound of the distribution
                    expected value of the normal distribution
        interval:   list of 2 floats, default None
                    interval on which to plot the distribution 
        Returns:
        --------
        x:          numpy 1d-array
                    interval spanned by x (input)
        y:          numpy 1d-array        
                    underlying distribution function of x
        """
        if interval is not None:
            x = np.arange( *interval )
        else:
            x = np.arange( a, b+(b-a)/100, (b-a)/100)
        n = abs(b-a)

        phi = np.zeros( x.shape)
        for i in range( phi.shape[0]):
            if x[i] >= a and x[i] <= b:
                phi[i] = 1/n
        return x, phi 


    def lognormal( mu=0, sigma=1, interval=None):
        """
        sample the lognormal distribution based
        on its parameters on the given interval
        Parameters:
        -----------
        mu:         float, default 0
                    mu parameter of the distribution
        sigma:      float, default 1 
                    sigma paramter of the distribution 
                    expected value of the normal distribution
        interval:   list of 2 floats, default None
                    interval on which to plot the distribution 
        Returns:
        --------
        x:          numpy 1d-array
                    interval spanned by x (input)
        y:          numpy 1d-array        
                    underlying distribution function of x
        """
        if interval is not None:
            x = np.arange( *interval )
        else:
            increment = 4*sigma/200
            #x = np.arange( 0+increment, 5*mu + 10*sigma+increment, increment ) 
            x = np.arange( 0.01, 5*(1+mu**3), 1/100 ) #should prolly be something with e^
        y = 1/(sqrt( 2*pi)*(sigma*x)) * exp( - (np.log( x) -  mu)**2/(2*sigma**2) )
        return x, y


    def levy( mu=0, gamma=1, interval=None):
        """
        samples n values out of the levy distribution
        with the given parameters
        Parameters:
        -----------
        mu:         float, default 0
                    mu parameter (start position)
        gamma:      float, default 1
                    variance parameter (sigma^2)
        interval:   list three floats, default None
                    interval of sample defined as [start,stop,step]
        Returns:
        --------
        x:          numpy 1d-array
                    interval spanned by x (input)
        y:          numpy 1d-array        
                    underlying distribution function of x
        """
        if interval is not None:
            x = np.arange( *interval )
        else:
            x = np.arange( mu+0.01, 8*gamma+0.01 +(8*gamma-mu)/100, (8*gamma-mu)/200 ) 
        y = np.sqrt( gamma/(2*pi) ) * 1/( x- mu)**(3/2) * exp( - gamma/(2*(x-mu) ) )
        return x, y


    def t_distribution( nu, interval ):
        """
        Return two arrays to plot the t-distirbution based on its parameter
        Parameters:
        -----------
        nu          float
                    degrees of freedom for the distributino 
        interval:   list three floats, default None
                    interval of sample defined as [start,stop,step]
        Returns:
        --------
        x:          numpy nd-array
                    sample array of the interval in 100 equidistant steps
        y:          numpy nd-array
                    f(x) of the interval above 
        """ 
        if interval is not None:
            x = np.arange( *interval )
        else:
            x = np.arange( *interval )
        if nu > 320:
            print( "can't compute the gamma function for large nu (>320)\nReturning approximation of t-distribution")
            y = np.exp( -x**2/2)/ np.sqrt( 2*pi)
        else:
            y = (gamma( ( nu+1 )/2 ) / (np.sqrt( pi*nu)* gamma( nu/2) ) 
                    * (1+ x**2/nu)**(-(nu+1)/2) )
        return x, y


    def chi( k, interval=None):
        """
        Return two arrays to plot the t-distirbution based on its parameter
        Parameters:
        -----------
        k:          int
                    parameter of the chi distribution
        interval:   list three floats, default None
                    interval of sample defined as [start,stop,step]
        Returns:
        --------
        x:          numpy nd-array
                    sample array of the interval in 100 equidistant steps
        y:          numpy nd-array
                    f(x) of the interval above 
        """
        if interval is not None:
            x = np.arange( *interval )
        else:
            x = np.arange( -3.5, 3.57, 7/100 ) #TODO find a fitting interval
        y = x**(k-1) * np.exp( -(x**2)/2) / (2**(k/2-1) *gamma(k/2) )
        return x, y
        

class data_binning:
    """
    compute the bins for data which is sampled in a 1d-array.
    This merely counts the number of samples within the bounds of the bin,
    defined by the interval sampled in the array
    """
    def bin_data( data, n_bins=None ):
        """
        Count the data and return the number of samples per bin
        If the number if n_bins is not specified it will automatically be computed
        Parameters:
        -----------
        data:       numpy 1d-array
                    given data
        n_bins:     int, default None
                    How many bins the data should be put into
                    Computes a sensitive default value if not specified

        Returns:
        --------
        count:      numpy nd-array
                    number of samples in the given bin
        center:     numpy nd-array
                    center value of the given bin
        width:      float
                    width of the bin
                    bin start/end is center -/+ width/2
        """
        n_samples = data.shape[0]

        # automatical choice for the number of bins
        if n_bins is None:
            if n_samples < 100:
                print( 'too few data samples present, returning un-binned data' )
                return [data.copy()]
            else:
                # Square-root choice
                n_bins = int(np.ceil(np.sqrt(n_sample)))
                
        inspected_values = data[:].flatten()
        sorting          = np.argsort( inspected_values)
        data             = data[ sorting ] 
        data_bins   = []
        lower_bound = inspected_values.min()
        upper_bound = inspected_values.max() - lower_bound
        stepsize    = upper_bound/ n_bins

        center      = (0.5 + np.arange(n_bins))*stepsize+lower_bound
        width       = stepsize
        count       = np.zeros(n_bins)
        previous_sample = 0
        for i_bin in range( 0,n_bins-1):
            for j in range( previous_sample, n_samples):
                if data[ j ] > center[i_bin]+0.5*stepsize:
                    count[i_bin] = j-previous_sample
                    previous_sample = j 
                    break
        count[-1] = n_samples-count[0:-1].sum()
        return count, center, width


    def rebin_data( count, center, n_bins, width=None):
        """
        Put the binned data into n_bins other bins.
        n_bins should be smaller than the number of previous bins, i.e. len(count)
        Parameters:
        -----------
        count:      numpy 1d-array 
                    binned data, consisting of the number of samples per bin
        n_bins:     int
                    number of bins after re-binning. n_bins < len( count) !
        center:     numpy 1d-array
                    center of the bin, is used to accurately compute bins
                    assumes a uniform distribution of samples in each bin
        width:      float or numpy 1d-array, default None
                    width of each bin, if None is given then each is calculated
                    The median width is assumed for the first and last bin
                    NOT YET IMPLEMENTED, assumes equidistant bins
        Returns:
        --------
        count:      numpy 1d-array
                    new binned count
        center:     numpy 1d-array
                    new center of the bins
        width:      float or numpy 1d-array
                    new width of the bins
        """
        bin_length = len( count)/n_bins
        bin_overlap = bin_length % 1
        new_count = np.zeros( n_bins)
        remainder = 0.
        for i in range( n_bins -1):
            new_count[i] = count[ floor(i* bin_length)] * (1-remainder ) 
            new_count[i] += sum( count[ floor(i* bin_length)+1 : floor( (i+1) * bin_length) ] ) 
            remainder = round(remainder + bin_overlap, 15) % 1
            new_count[i] +=  count[floor( (i+1) * bin_length)] * remainder 
        i +=1
        new_count[ -1] =  count[ floor(i* bin_length) ]  * (1-remainder )
        new_count[ -1] += sum( count[ floor(i* bin_length)+1 :] )

        #assume equidistant bins
        center_incr = center[1]-center[0] 
        center = np.arange( center[0], center[-1]+ center_incr, center_incr * bin_length )[:n_bins] + (bin_length/2 - center_incr/2)
        if width is not None:
            if isinstance( width, np.ndarray ):
                width = (bin_length * width)[:n_bins] 
            else:
                width = bin_length * width
        else:
            width = None
        return new_count, center, width




    def percentile( alpha, count, center=None, width=None ):
        """
        Compute the percentile of the binned data

        Parameters:
        -----------
        alpha:      float
                    searched percentile value
        count:      numpy 1d-array
                    number of samples in the bins
        center:     numpy 1d-array, default None
                    center value of the bins
        width:      numpy 1d-array, default None
                    width of the bins

        """
        n = count.shape[0]
        if( center is None ):
            center = np.arange(n)
            
        # step 1: find k such that count.cumsum()[k] >= alpha*n
        # find the k-th bin in which the percentile lies
        cum = np.cumsum(count) 
        k   = 0
        N   = count.sum()*alpha
        while(cum[k] < N and k<n ):
            k += 1
        # step 2: if the data is discrete, then we are good
        if( width is None ):
            ## Debug only
            #print('k=%d, N=%d, left sum: %d, at k: %d, right sum: %d' % (k, N, n_left, count[k], n_right ))
            return center[k]
        
        # more computations needed: estimate the position within the bin:
        n_left  = count[0:k].sum()
        n_right = count[(k+1):].sum()
        theta = (N-n_left)/count[k]
        #print('k=%d, N=%d, left sum: %d, at k: %d, right sum: %d' % (k, N, n_left, count[k], n_right ))
        #print('theta=%f' % theta)
        return center[k] + (theta - 0.5) * width[k]


    def compute_relative_frequency( count ):
        """
        Compute the relative frequency of the data in their bins
        It is assumed that the data in each bin contains n-row samples
        The mean value of each bin will be 
        Parameters:
        -----------
        count:      numpy 1d-array
                    containing the count for each bin
        Returns:
        --------
        frequency:  numpy-1d array
                    relative frequency for each bin
        """
        frequency = count/count.sum()
        return frequency

    def compute_cumulative_frequency( count ):
        """
        Compute the cumulative frequency of binned data
        It is assumed that the data in each bin contains n-row samples
        The mean value of each bin will be 
        Parameters:
        -----------
        count:      numpy 1d-array
                    containing the count for each bin
        Returns:
        --------
        cumulative_frequency:   numpy-1d array
                                relative frequency for each bin
        """
        frequency = data_binning.compute_relative_frequency( count )
        cumulative_frequency = np.cumsum(frequency)
        return cumulative_frequency

