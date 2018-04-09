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

# Scaling data between values
from sklearn.preprocessing import MinMaxScaler

# Neural Network model
from keras.models import Sequential

# Type of neural network layer
from keras.layers import Dense

# Argument Parsing
import argparse

# Utility
import sys # exiting properly
import code # Debugging

###############################################################################
# Directories

FIGS_DIR = './figs/'
DATASETS_DIR = './datasets/'

###############################################################################
# Global Functions


def parse_arguments():
    """ Parse options to do one exercise at a time. """

    parser = argparse.ArgumentParser()

    # Specify if one wants to perform the persistence forecast
    parser.add_argument(
        '-p',
        '--persistence-forecast',
        action='store_true',
        help='if set, perform persistence forecast'
    )

    # Specify if one wants to look for the best epoch value
    parser.add_argument(
        '-e',
        '--epoch',
        action='store_true',
        help='if set, look for the best epoch value'
    )

    return parser

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

def persistence_forecast(dataset):
    """ Make a persistence forecast on the test dataset and measure performance
    """

    # Get first half
    training = dataset.iloc[:len(dataset.values)//2]

    # Get second half
    test = dataset.iloc[len(dataset.values)//2:]

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

def test_model(dataset, repeats, lag, hidden_layers, epoch):
    """ Return a list of 'repeats' mean absolute error with:
        - A difference degree of 'lag' to make the time series stationary (i.e.
        same mean, variance, covariance, and no seasonal trends)
        - One hidden layer for each value in 'hidden_layers', each containing
        value neurons
        - 'epoch' epochs of training
    """

    ###########################################################################

    def add_outputs(df, lag):
        """ Add a column to specify the expected output for each input.
            Example:
                |x|
                |y|
                |z|
            Becomes:
                |NaN|x|
                |x|y|
                |y|z|
        """

        df.columns = ['Output']
        shifted = df.shift(lag)
        shifted.columns = ['Input']

        return pd.concat([shifted, df], axis=lag)[lag:]

    ###########################################################################

    def scale(df, lower=-1, upper=1):
        """ Scale between 1 and -1. """

        scaler = MinMaxScaler(feature_range=(lower, upper))
        scaler = scaler.fit(df)

        # Replace values with the scaled dataframe
        df[['Input', 'Output']] = scaler.transform(df)

        return df

    ###########################################################################

    def create_model(training, layers, activation='relu',
                     loss='mean_absolute_error', optimizer='adam'):
        """ Creates a Sequential neural network with one layer for each value
            in 'layers', each containing value neurons.
            The default activation method used by default is the rectified linear unit
            (relu).
            The default optimizer used is adam.
        """

        model = Sequential()

        # Add the first layer with the input dimension
        model.add(Dense(layers[0], activation=activation, input_dim=1))

        for number_of_neurons in layers[1:]:
            model.add(Dense(number_of_neurons, activation=activation))

        # Specify that there is only one returned value
        model.add(Dense(1, activation=activation))

        model.compile(loss=loss, optimizer=optimizer)

        return model

    ###########################################################################

    df = dataset.copy()

    # Make time series stationary
    df = dataset.diff(periods=lag)[lag:] # Remove 'lag' NaN values

    # Add one column to associate inputs with outputs
    df = add_outputs(df, lag)

    # Scale dataframe between -1 and 1
    df = scale(df)

    # Split data into training and test set
    training = df.iloc[:len(df.values)//2] # first half
    test = df.iloc[len(df.values)//2:] # Second half

    # Now that the data is properly formatted it is possible to run the
    # experiments
    for experiment_number in range(repeats):
        model = create_model(training, hidden_layers)

        # Training
        model.fit(
            training[[0]],
            training[[1]],
            epochs=epoch
        )

        # Forecasting: TODO
        output = model.predict(test)
        predictions = []

###############################################################################
# Main


if __name__ == '__main__':

    # Argument parsing
    parser = parse_arguments()
    args = parser.parse_args()

    # If no arguments are given, do everything
    if not (args.persistence_forecast or args.epoch):
        args.persistence_forecast = True
        args.epoch = True

    dataset = load_dataset()

    # Change plotting options to make plots prettier
    set_matplotlib_parameters()

    # Perform persistence forecast
    if args.persistence_forecast:
        persistence_forecast(dataset)

    # Look for best epoch value
    if args.epoch:
        # Parameters
        repeats = 20 # Number of times the experiment will be repeated
        lag = 1 # Difference order to make the time series stationary
        hidden_layers = [20] # One layer with one neuron
        epochs = [250, 500, 750, 1000, 2000]

        results = pd.DataFrame()

        for epoch in epochs:
            results[str(epoch)] = test_model(
                dataset,
                repeats,
                lag,
                hidden_layers,
                epoch
            )
