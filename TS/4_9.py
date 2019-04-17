from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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
# distance = m._dtw_distance(amplitude_a, amplitude_b)

# fig = plt.figure(figsize=(12,4))
# plt.plot(time, amplitude_a, label='A')
# plt.plot(time, amplitude_b, label='B')
# plt.title('DTW distance between A and B is %.2f' % distance)
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














# Plot the yearly temperatures for Norway, Finland, Singapore and Cambodia. Use DTW to measure
# the similarities between the temperature data of these countries and reflect on the results


# temperatures per country
tpc = pd.read_csv("../climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")

### Decide where to start with the dates
# wer = tpc[tpc['Country'] == "Singapore"].reset_index()                # Get data of country
# print(wer.info())
# isnull = wer[wer['AverageTemperature'].isnull()]
# print(isnull[-5:])

# plt.clf()
# plt.plot(wer['AverageTemperature'])
# plt.show()
# exit()


def dateToInt(date):
    [year, month, _] = date.split("-")
    return int(year) * 12 + int(month)

countries = ["Norway", "Finland", "Singapore", "Cambodia"]
dataPerCountry = {}
startingDate = "1863-01-01"
startingMonth = dateToInt(startingDate)
nYears = 150
print("startingMonth: %s" % startingMonth)

plt.clf()
for country in countries:
    data = tpc[tpc['Country'] == country]                # Get data of country
    data = data.reset_index()
    data = data[['dt', 'AverageTemperature']]            # Drop unnecessary columns
    data = data[data['AverageTemperature'].notnull()]    # Drop columns with NaN values
    data['months'] = data['dt'].transform(lambda date : dateToInt(date))    # Transform datestring into a number
    print("%s : %d rows, from %s to %s" % (country.rjust(10), data.count()[0], data.iloc[0]['dt'], data.iloc[-1]['dt']))
    
    index = data['months'].between(startingMonth, startingMonth + nYears * 12)   # Get data for x years, starting at 1863-01-01
    yAxis = data[index] 
    
    yAxis = yAxis['AverageTemperature'].astype('float')            # Transform temperature to float
    yAxis = yAxis.values

    # yAxis -= yAxis.mean()                            # Substract mean for DTW
    dataPerCountry[country] = yAxis                  # Store data of country in this map
    plt.plot(yAxis, label=country, linewidth=1.0)    # Plot the data

plt.legend()
plt.title("Temperature over %d years (%d datapoints)" % (nYears, nYears * 12))
# plt.show()


#####################################
########## Print DTW table ##########
#####################################

# dtw = KnnDtw()
# print("\n# Table with minimal DTW distance")
# HeaderRow = "DISTANCE ".ljust(10)
# for i1, c1 in enumerate(countries):
#     HeaderRow += c1.ljust(10)
# print(HeaderRow)

# for i1, c1 in enumerate(countries):
#     Row = (c1 + " ").rjust(10)
#     for i2, c2 in enumerate(countries):
#         s1 = dataPerCountry[c1]
#         s2 = dataPerCountry[c2]
#         distance, cost = dtw._dtw_distance(s1, s2)
#         Row += str(int(distance)).ljust(10)
#     print(Row)



###########################################
########## B) Dickey Fuller Test ##########
###########################################

print('\nDickey-Fuller Test:')
for country in countries:    
    dftest = adfuller(dataPerCountry[country], autolag='AIC')
    pvalue = dftest[1]
    print(("  P-value for %s" % country).ljust(25), "%0.4f%%" % (pvalue * 100))
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value
    # print(dfoutput)


###########################################
###########################################
###########################################

plt.clf()
# for country in countries:
for country in countries:
    y1 = dataPerCountry[country]
    y1 -= y1.mean()
    y2 = [np.cbrt(y) for y in y1]
    # plt.plot(y1, label="%s y1" % country)
    plt.plot(y2, label="%s y2" % country)

plt.legend()
# plt.show()


for country in countries:
    print(country)
    y1 = dataPerCountry[country]
    decomposition = seasonal_decompose(y1, freq=900)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.clf()
    plt.subplot(411)
    plt.plot(y1, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

