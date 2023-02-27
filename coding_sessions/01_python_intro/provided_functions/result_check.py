import sys
import numpy as np

def list_result(x, y):
    """
    an assertion error is thrown if the part is not implemented correctly
    displays the error, terminates the program
    """
    error_msg = 'Lists do not match, please correct!\n desired list:\t {}\n entered list:\t {}\n'.format( y, x)
    if isinstance( y, list) and isinstance( x, list):
        assert not( x is y), "Don't cheat, solve the exercise!"
    try: #if lists are equally long
        if not x == y:
            raise Exception( error_msg )
    except:
        raise Exception( error_msg )
    print( 'Part correctly solved.') 
