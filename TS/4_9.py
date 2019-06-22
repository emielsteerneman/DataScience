from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

import sys
import collections
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fftpack
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis
from scipy.stats import mode
from scipy.spatial.distance import squareform

plt.style.use('bmh')

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter(action='ignore', category=FutureWarning)

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function 
            
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
            
        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
        self.x = x
        self.l = l
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = 1e10 * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])
   
        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
                # print("i=%d j=%d cost=%0.4g" % (i, j, cost[i, j]))

        # print("Cost matrix:")
        # print(self._print_cost_matrix(cost))

        # Return DTW distance given window 
        return cost[-1, -1], cost
    
    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        
        y : array of shape [n_samples, n_timepoints]
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            p = ProgressBar(np.shape(dm)[0])
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
                    p.animate(dm_count)
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            dm_size = x_s[0]*y_s[0]
            
            p = ProgressBar(dm_size)
        
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j], _ = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)
        
            return dm
        
    def _print_cost_matrix(self, cost):
        [i, j] = cost.shape
        print("i=%d, j=%d" % (i, j))
        cost[0][0] = 1
        cost[0][1] = 1
        for row in cost:
            r = "  "
            for c in row:
                r += ("%.4g" % c).rjust(11)
            print(r)

    def predict(self, x):
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels 
              (2) the knn label count probability
        """
        
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r', self, end="", flush=True)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)





######## Test example ########
# time = np.linspace(0,20,1000)
# amplitude_a = 5*np.sin(time)
# amplitude_b = 3*np.sin(time + 1)

# m = KnnDtw()
# distance, cost = m._dtw_distance(amplitude_a, amplitude_b)
# fig = plt.figure(figsize=(12,4))
# plt.plot(time, amplitude_a, label='A')
# plt.plot(time, amplitude_b, label='B')
# plt.title("DTW distance between A and B is %.2f" % distance)
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.legend()
# plt.show()
##############################

# def plotSignalAndDTW(s1, s2, x = None):
#     if x is None:
#         maxLen = max(len(s1), len(s2))
#         x = np.arange(0, maxLen, 1)
#     print(x[-1])

#     m = KnnDtw()
#     distance, cost = m._dtw_distance(s1, s2)

#     # Find shortest path
#     [M, N] = np.subtract(cost.shape, (1, 1)) # [Rows, Columns]
#     path = []
#     while M != 0 or N != 0:
#         costs = [cost[M-1, N-1], cost[M-1, N], cost[M, N-1]]
#         lowest = np.argmin(costs)
#         [M, N] = [[M-1, N-1], [M-1, N], [M, N-1]][lowest]
#         # print(costs, lowest, [M, N])
#         path.append([M, N])

#     path.reverse()

#     [p1, p2] = list(map(list, zip(*path)))

#     _p1 = list(zip(p1, s1[p1]))
#     _p2 = list(zip(p2, s2[p2]))
#     _p = list(zip(_p1, _p2))

#     plt.clf()
#     plt.plot(x, s1)
#     plt.plot(x, s2)
#     for point in _p:
#         [[x0, y0], [x1, y1]] = point
#         plt.plot([x0, x1], [y0, y1], color="black", linewidth=0.1)
#     plt.title("Distance: %0.4f" % distance)
#     plt.show()


#########################################################################################
#########################################################################################











#######################
########## A ##########
#######################

# Plot the yearly temperatures for Norway, Finland, Singapore and Cambodia. Use DTW to measure
# the similarities between the temperature data of these countries and reflect on the results

def dateToInt(date):
    [year, month, _] = date.split("-")
    return int(year) * 12 + int(month)

countries = ["Norway", "Finland", "Singapore", "Cambodia"]
tpc = pd.read_csv("../climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")

########## Figuring out at what date the last NaN occurs in AverageTemperature ##########
print("Figuring out appropriate starting date..")
# All rows of the selected countries where the average temperature is NaN
indicesNaN = tpc["Country"].isin(countries) & tpc["AverageTemperature"].isna()
groups = tpc[indicesNaN].groupby("Country")
dates = []
for country, group in groups:
    print(country.rjust(12), "|", group["dt"].values[-2])
    dates.append(group["dt"].values[-2]) # Get the last date where a NaN value occurs in AverageTemperature
dates.sort(reverse=True)
startingDate = dates[0]
startingMonth = dateToInt(startingDate) + 1
print("Last NaN value for AverageTemperature in %s \n" % startingDate)


dataPerCountry = {}
nYears = 100

plt.clf()
for country in countries:
    data = tpc[tpc['Country'] == country]                # Get data of country
    data = data.reset_index()
    data = data[['dt', 'AverageTemperature']]            # Drop unnecessary columns
    data['months'] = data['dt'].transform(lambda date : dateToInt(date))    # Transform datestring into a number
    print("%s : contains %d rows, from %s to %s" % (country.rjust(10), data.count()[0], data.iloc[0]['dt'], data.iloc[-1]['dt']))
    
    index = data['months'].between(startingMonth, startingMonth + nYears * 12 - 1)   # Get data for x years, starting at 1863-01-01
    yAxis = data[index] 
    
    yAxis = yAxis['AverageTemperature'].astype('float')            # Transform temperature to float
    yAxis = yAxis.values
    # yAxis -= yAxis.mean()                            # Substract mean for DTW
    dataPerCountry[country] = yAxis                  # Store data of country in this map
    plt.plot(yAxis, label=country, linewidth=1.0)    # Plot the data

plt.legend()
plt.title("Temperature over %d years (%d datapoints)" % (nYears, nYears * 12))
plt.tight_layout()
# plt.show()

########## Print DTW table ##########
# dtw = KnnDtw()
# print("\nTable with minimal DTW distance")
# HeaderRow = "DISTANCE ".ljust(10)
# for i1, c1 in enumerate(countries):
#     HeaderRow += c1.ljust(10)
# print(HeaderRow)

# for i1, c1 in enumerate(countries):
#     Row = (c1 + " ").rjust(10)
#     for i2, c2 in enumerate(countries):
#         distance, cost = dtw._dtw_distance(dataPerCountry[c1], dataPerCountry[c2])
#         Row += str(int(distance)).ljust(10)
#     print(Row)



###########################################
########## B) Dickey Fuller Test ##########
###########################################

# print('\nDickey-Fuller Test:')
# for country in countries:    
#     dftest = adfuller(dataPerCountry[country], autolag='AIC')
#     adf = dftest[0]
#     pvalue = dftest[1]
    
#     print(country.rjust(12), end="   ")
#     print("ADF=%0.3f" % adf, end="   ")
#     print("p-value=%0.3f" % pvalue, end="   ")
#     print("Critical Values : ", end="")
#     for k, v in dftest[4].items():
#         print("%s=%0.10f" % (k, v), end="   ")
#     print()


########################
########## C) ##########
########################

# Temperature and weather data includes seasons, day and night temperature changes as well as global
# warming. This seasonality and the slow trends (such as global warming) can be removed by differencing 
# and decomposition techniques. Apply these techniques from the tutorial to de-trend the data and remove
# seasonality. Again apply DTW on your newly obtained processed data and reflect on the results

## Apply differencing ###
# plt.clf()
# differencingPerCountry = {}
# for country in countries:
#     data = dataPerCountry[country]
#     diff = data - np.roll(data, 1)
#     differencingPerCountry[country] = diff
#     plt.plot(diff, label=country)
# plt.legend()
# plt.title("Differencing per month")
# plt.tight_layout()
# plt.show()

### Apply decomposition ###
# plt.clf()
seasonPerCountry = {}
trendPerCountry = {}
for country in countries:
    data = pd.Series(dataPerCountry[country])
    decomp = seasonal_decompose(data, freq=12)
    
    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid
    
    # Fix NaN at the beginning and end of trend
    trend[:6] = [trend[6]] * 6
    trend[len(trend)-6:] = [trend[len(trend)-7]] * 6

    seasonPerCountry[country] = seasonal
    trendPerCountry[country] = trend
    
#     plt.subplot(211)
#     plt.plot(seasonal, label=country)
#     plt.subplot(212)
#     plt.plot(trend, label=country)

# plt.subplot(211)
# plt.legend()
# plt.title("Seasonality")
# plt.subplot(212)
# plt.legend()
# plt.title("Trend")
# plt.tight_layout()
# plt.show()

### Remove trend and seasonality from data ###
# plt.clf()
stationaryDataPerCountry = {}
for country in countries:
    data = dataPerCountry[country]
    newData = data - seasonPerCountry[country] - trendPerCountry[country] # Subtract trend and seasonality
    stationaryDataPerCountry[country] = newData
#     plt.plot(newData, label=country)
# plt.legend()
# plt.title("Without trend and seasonality over 15 years")
# plt.tight_layout()
# plt.show()

### Print DTW table ###
# dtw = KnnDtw()
# print("\nTable with minimal DTW distance")
# HeaderRow = "DISTANCE ".ljust(10)
# for i1, c1 in enumerate(countries):
#     HeaderRow += c1.ljust(10)
# print(HeaderRow)

# for i1, c1 in enumerate(countries):
#     Row = (c1 + " ").rjust(10)
#     for i2, c2 in enumerate(countries):
#         distance, cost = dtw._dtw_distance(stationaryDataPerCountry[c1], stationaryDataPerCountry[c2])
#         Row += str(int(distance)).ljust(10)
#     print(Row)




#######################
########## D ##########
#######################
# Read the section on Forecasting models in the tutorial and apply the AR, AM and ARIMA model on
# the temperature data of one of the four countries. Reflect on the results of the models on the data. Be
# clear in your methodology and explain which values for p, q and d you use based on the ACF and
# PACF plots.

country = countries[0]
dataStationary     = stationaryDataPerCountry[country][:-12*5]
dataStationaryTest = stationaryDataPerCountry[country][-12*5:]
data     = dataPerCountry[country][:-12*5]
dataTest = dataPerCountry[country][-12*5:]

print("\nApplying forecasting models to %s" % country)


# [19] Once we have got the stationary time series, we must answer two primary questions:
# Q1. Is it an AR or MA process?
# Q2. What order of AR or MA process do we need to use?

# [19]
# p : Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
# q : Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
# d : Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.

lag_acf = acf(dataStationary, nlags=20)
lag_pacf = pacf(dataStationary, nlags=20, method='ols')

plt.clf()
#[19] Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataStationary)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataStationary)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#[19] Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataStationary)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataStationary)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

# [19] p – The lag value where the PACF chart crosses the upper confidence interval for the first time
# [19] q – The lag value where the  ACF chart crosses the upper confidence interval for the first time
p = 3
q = 2

### AR MODEL ###
# [19]
try:
    model = ARIMA(data, order=(p, 1, 0))
    results_AR = model.fit(disp=-1)  
    plt.clf()
    plt.plot(dataStationary)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('AR 15 years - RSS: %.4f'% sum((results_AR.fittedvalues - dataStationary[:-1])**2))
    plt.show()
except:
    print("Could not create AR model")

### MA MODEL ###
# [19]
try:
    model = ARIMA(data, order=(0, 1, q))  
    results_MA = model.fit(disp=-1)  
    plt.clf()
    plt.plot(dataStationary)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('MA 15 years - RSS: %.4f'% sum((results_MA.fittedvalues - dataStationary[:-1])**2))
    plt.show()
except:
    print("Could not create MA model")

### ARIMA MODEL ###
# [19]
try:
    model = ARIMA(data, order=(p, 1, q))  
    results_ARIMA = model.fit(disp=-1)  
    plt.clf()
    plt.plot(dataStationary)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('ARIMA 15 years - RSS: %.4f'% sum((results_ARIMA.fittedvalues - dataStationary[:-1])**2))
    plt.show()
except:
    print("Could not create ARIMA model")


#######################
########## E ##########
#######################
# Select one of the models from the question above to make a temperature forecast (using for example
# the forecast() and predict() methods from the models) for the next seven days for your country
# of choice. You can take the next seven days from the last date present in the dataset of that country.
# Reflect on the quality of the prediction.

forecast, stderr, conf_int = results_ARIMA.forecast(steps=len(dataTest))
plt.clf()
plt.plot(dataTest, label="Actual")
plt.plot(forecast, label="Prediction", color="red")
plt.title("Forecast vs actual temperatures over 5 years")
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.show()






exit()