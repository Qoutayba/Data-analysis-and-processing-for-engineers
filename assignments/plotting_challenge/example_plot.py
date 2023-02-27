import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re


## The data can be found online under this link (with daily updates): 
# oil: https://www.nasdaq.com/market-activity/commodities/bz%3Anmx/historical
# gas: https://www.nasdaq.com/market-activity/commodities/ng%3Anmx/historical

## Data loading and data overview
price = 'Oil' 
if price.lower() == 'oil':
    stock_data = pd.read_csv( 'data/BZ-NMX_oil.csv') 
elif price.lower() == 'gas':
    stock_data = pd.read_csv( 'data/NG-NMX_gas.csv') 
else:
    raise Exception( 'Please specify stock-index of available data' )
print( '#########################################')
print( '#### First few rows of the dataframe ####')
print( stock_data.head())
print( '#### Last few rows of the dataframe  ####')
print( stock_data.tail() )
print( '#########################################')

## Process csv data to be usable for matplotlib
dates = np.array( stock_data['Date'])
for i in range( len( dates) ):
    #regular expression for reformatting each string
    dates[i] = re.sub( '(\d*)\/(\d*)\/(\d*)', r'\3/\1/\2', dates[i]).replace('/','-' )
stock_data['Date'] = dates

## Plotting of data
fig, ax = plt.subplots()
ax.plot( stock_data['Date'], stock_data['Close/Last'] ) 

## Plot decorators
ax.xaxis.set_major_locator( mdates.AutoDateLocator())
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
fig.autofmt_xdate()
ax.set( xlabel='Date [yyyy-mm-dd]', ylabel='Price [$]', 
        title='Historical Price development of {}'.format( price ) )
ax.grid()
plt.show()
