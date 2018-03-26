# Machine Learning

The goal of this repository is to showcase some machine learning applications.
The main focus will be time series forecasting using Python3.

## Dataset

The dataset that is going to be used is [Mean summer temperature (in deg. C),
1781-1988]
(https://datamarket.com/data/set/22m7/mean-summer-temperature-in-deg-c-1781-1988#!ds=22m7&display=line).
I found it in the [Time Series Data Library](https://datamarket.com/data/list/?q=provider:tsdl),
a compilation of time series made by Rob Hyndman, Professor of Statistics at
Monash University, Australia.

This is what the dataset looks like:

![Dataset](./figs/dataset.png)

This dataset only contains 208 values but I've chosen to use it because a simple
example will be easier to understand.

## Loading the dataset

The data is going to be formatted in a pandas dataframe.
[Pandas](https://pandas.pydata.org/) is a powerful library that contains an
incredible amount of built-in functions.
In order to be able to use it on a Debian-based distribution, one needs to run 
`apt-get install python3-pandas`.

The dataset is saved inside a `.csv`, which means that it is possible to load it
using the `read_csv` function.
The code is the following:

```Python3
import pandas as pd

def parser(date):
    """ Takes a year as a parameter and returns a datetime """
    
    return pd.datetime.strptime(date, '%Y')

dataframe = pd.read_csv(
    DATASETS_DIR + mean_summer,  # where the dataset file is
    names=['Year', 'Temperature'],  # column names
    index_col=0,  # column to be used as index - 'year' in this case
    skiprows=[0, 210], # skip column names, and the dataset's name
    parse_dates=[0],  # parse first column into dates
    date_parser=parser  # function used to parse dates
)
```

Some lines need to be removed, which is why the `skiprows` paremeter is used.
Aside from that, it us useful to parse dates into a `pandas.Datetime` object to
be able to perform operations on dates.