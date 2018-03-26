#!/usr/bin/python3.5

###############################################################################
# Imports

# Data Formatting
import pandas as pd

###############################################################################
# Directories

FIGS_DIR = './figs/'
DATASETS_DIR = './datasets/'

###############################################################################
# Global Functions


def load_dataset(mean_summer='mean-summer-temperature-in-deg-c.csv'):
    """ Loads the 'Mean Summer Temperature' dataset into a pandas DataFrame """

    ###########################################################################

    def parser(date):
        """ Takes a year as a parameter and returns a datetime """

        return pd.datetime.strptime(date, '%Y')

    ###########################################################################

    return pd.read_csv(
        DATASETS_DIR + mean_summer,  # where the dataset file is
        names=['Year', 'Temperature'],  # column names
        index_col=0,  # column to be used as index - 'year' in this case
        skiprows=[0, 210], # skip column names, and the dataset's name
        parse_dates=[0],  # parse first column into dates
        date_parser=parser # function used to parse dates
    )

###############################################################################
# Main


if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset)
