#!/usr/bin/python3.5

###############################################################################
# Imports

# Data Formatting
import pandas as pd

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt

# Regression Metric
from sklearn.metrics import mean_absolute_error

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

def set_matplotlib_parameters():
    # Plot size to 14" x 7"
    matplotlib.rc('figure', figsize=(14, 7))

    # Font size to 14
    matplotlib.rc('font', size=14)

    # Do not display top and right frame lines
    matplotlib.rc('axes.spines', top=False, right=False)

    # Remove grid lines
    matplotlib.rc('axes', grid=False)

    # Set backgound color to white
    matplotlib.rc('axes', facecolor='white')

    # Set pyplot style
    plt.style.use('seaborn-dark-palette')

###############################################################################
# Forecasting Methods

def persistence_forecast(training, test):
    """ Make a persistence forecast on the test dataset and measure performance
    """

    # Last value that has been observed. It is initialized to the last value of
    # the training set because the first value of the test set will be equal to
    # the last value of the training set.
    last_value = training.iloc[-1]

    # List that will contain a prediction for every value in test
    predictions = pd.DataFrame(columns=['Temperature'])

    # For each value in the training set
    for index, value in test.iterrows():
        # Append our prediction to the 'predictions' list
        predictions.loc[index] = last_value

        # Modify the last value to be the current one
        last_value = value

    # Compute mean absolute error
    error = mean_absolute_error(test, predictions)

    # Plot of the difference
    plt.plot(test, label='test')
    plt.plot(predictions, label='predictions')

    # Plot options
    plt.xlabel('Year')
    plt.ylabel('Temperature')
    plt.title('Persistence forecast of the mean temperature.'
              'Mean absolute error: {}'.format('%.2f' % error))
    plt.gca().legend(loc='best')
    plt.tight_layout()

    # Draw plot
    plt.savefig(FIGS_DIR + 'persistence.png')

###############################################################################
# Main


if __name__ == '__main__':
    dataset = load_dataset()

    # Get first half
    training = dataset.iloc[:len(dataset.values)//2]

    # Get second half
    test = dataset.iloc[len(dataset.values)//2:]

    # Change plotting options to make plots prettier
    set_matplotlib_parameters()

    # Perform persistence forecast
    persistence_forecast(training, test)
