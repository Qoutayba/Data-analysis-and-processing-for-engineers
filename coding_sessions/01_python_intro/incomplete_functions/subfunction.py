def add(a,b):
    return a+b


def multiply_and_divide( a,b):
    """
    Multiply a and b and divide a by b
    Care: Function does not work for b=0
    Parameters:
    -----------
    a:          int or float 
                first number, nominator
    b:          int or float 
                second number, denominator, can not be 0
    Returns:
    --------
    prod:    int or float
                product of a and b
    frac:   float
                fraction of a divided by b
                    
    """
    if b == 0:
        raise Exception( 'error in multiply_and_divide, can not divide by b=0!' )
    prod1 = a*b
    frac1 = a/b
    return prod1, frac1 #return both values


def oddly_weighted_sum( a, b, weight=1.618):
    """
    Compute the sum of two values and multiply by a weight
    Parameters:
    -----------
    a:          int or float 
                number used for summation
    b:          int or float 
                number used for summation
    weight:     float, default 1.618
                weight applied to the sum
    Returns:
    --------
    weighted_sum:   float
                    weighted sum of a and b
    """
    print( 'using the weight', weight, 'in "oddly_weighted_sum"')
    return (a+b)*weight


def product( *numbers):
    """
    Multiply all given numbers and return their product
    Parameters:
    -----------
    *numbers:   int or float values
                all numbers given for multiplication
    Returns:
    --------
    Product:    int or float
                product of all given numbers
    """
    result = 1
    for i in numbers:
        result = result *i
    return result
