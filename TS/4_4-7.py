import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fftpack
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis

### Load all files
print("Loading total_acc files...")


folder = "../UCI HAR Dataset/"

file_activity_labels = folder + "activity_labels.txt"
file_features = folder + "features.txt"
file_train_x = folder + "train/Inertial Signals/total_acc_x_train.txt"
file_train_y = folder + "train/Inertial Signals/total_acc_y_train.txt"
file_train_z = folder + "train/Inertial Signals/total_acc_z_train.txt"
file_test_x  = folder + "test/Inertial Signals/total_acc_x_test.txt" 
file_test_y  = folder + "test/Inertial Signals/total_acc_y_test.txt"
file_test_z  = folder + "test/Inertial Signals/total_acc_z_test.txt"

activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])
train_x = pd.read_csv(file_train_x, delimiter=" ", header=None, skipinitialspace=True)
train_y = pd.read_csv(file_train_y, delimiter=" ", header=None, skipinitialspace=True)
train_z = pd.read_csv(file_train_z, delimiter=" ", header=None, skipinitialspace=True)
test_x  = pd.read_csv(file_test_x,  delimiter=" ", header=None, skipinitialspace=True)
test_y  = pd.read_csv(file_test_y,  delimiter=" ", header=None, skipinitialspace=True)
test_z  = pd.read_csv(file_test_z,  delimiter=" ", header=None, skipinitialspace=True)

### Concatenate train-, and test-sets
print("Concatenating...")
x = pd.concat([train_x, test_x])
y = pd.concat([train_y, test_y])
z = pd.concat([train_z, test_z])



###############################
############# 4.4 #############
### For each axis, calculate the variance for each row and sum the variances
print("Calculating variances...")
x_var = x.apply(lambda row : row.var(), axis='columns').sum()
y_var = y.apply(lambda row : row.var(), axis='columns').sum()
z_var = z.apply(lambda row : row.var(), axis='columns').sum()
variances = [x_var, y_var, z_var]

print("  Variance x : %0.2f" % x_var)
print("  Variance y : %0.2f" % y_var)
print("  Variance z : %0.2f" % z_var)
print("Greatest variance : %s" % ['x', 'y', 'z'][np.argmax(variances)])


### Load body_acc files for axis with greatest variance
print()
print("Loading body_acc files...")
file_data = ['x', 'y', 'z'][np.argmax(variances)]
file_data_train = folder + "train/Inertial Signals/body_acc_" + file_data + "_train.txt"
file_labels_train = folder + "train/y_train.txt"
# file_data_test  = folder + "test/Inertial Signals/body_acc_" + file_data + "_test.txt"
# file_labels_test  = folder + "test/y_test.txt"

traindata = pd.read_csv(file_data_train, delimiter=" ", header=None, skipinitialspace=True)
trainlabels = pd.read_csv(file_labels_train, delimiter=" ", header=None, skipinitialspace=True)
# testdata  = pd.read_csv(file_data_test,  delimiter=" ", header=None, skipinitialspace=True)
# testlabels  = pd.read_csv(file_labels_test,  delimiter=" ", header=None, skipinitialspace=True)

trainlabels['label'] = trainlabels[0].transform(lambda c : activity_labels['activity'][c-1])
# testlabels['label']  = testlabels[0].transform(lambda c : activity_labels['activity'][c-1])

### Concatenate train-, and test-set
dataset  = traindata#pd.concat([traindata, testdata], ignore_index=True)
labelset = trainlabels#pd.concat([trainlabels, testlabels], ignore_index=True)

### Drop the last half of the columns to solve the overlap problem, allowing retrieval of the original signal
dataset = dataset.loc[:, :63]
### Convert the dataframe to a numpy array
raw_signal = dataset.values
### Flatten the 2D array to a 1D array by concatenating all the rows, effectively retrieving the original signal
raw_signal = raw_signal.flatten()
### Repeat labels so that we have a label per datapoint
raw_labels = np.repeat(labelset[0].values.flatten(), 64)

print()
print("Datapoints expected : 64 * %d = %d" % (len(dataset.index), 64 * len(dataset.index)))
print("Datapoints in raw signal        = %d" % raw_signal.size)
print("Labels for raw signal           = %d" % raw_labels.size)

### Concatenate signal and labels column-wise : "Again, couple the class labels (Y) with the raw data points (X)"
signal = np.vstack((raw_signal, raw_labels)).T

############# 4.4 #############
###############################



###############################
############# 4.5 #############
### 4.5 A
# "This means that you need to map the original data, for training , 7351 x 128 to the preprocessed data 7351 x 3 (for the features: mean, std, kurtosis)."

mean, std, kurt, labels = [], [], [], []

### Apply sliding window over raw signal
for i in range(0, raw_signal.size-64, 64):
	window = raw_signal[i:i+128]	# Grab window
	mean.append(window.mean())		# mean
	std.append(window.std())		# std
	kurt.append(kurtosis(window))	# kurtosis
	labels.append(raw_labels[i])	# label
### Create Pandas DataFrame for easy kde plotting
df = pd.DataFrame(np.vstack((mean, std, kurt, labels)).T, columns=["mean", "std", "kurt", "label"])
### Transform labels from int to string
df["label"] = df["label"].transform(lambda c : activity_labels['activity'][c-1])

### 4.5 B
groupsByLabel = df.groupby("label")
for feature in ["mean", "std", "kurt"]:
	plt.figure()
	groupsByLabel[feature].plot.kde()
	plt.legend(groupsByLabel.groups.keys())
	plt.title(feature)
	plt.show()
############# 4.5 #############
###############################




### 4.6

# ##### Testing Example #####
# print("\nRunning Fast Fourier Transform..")

# T = 1.0 / 80.0		# Samples per periods. Max frequency that can be recognized : 0.5 / T
# D = 4				# Total number of periods
# N = int(D / T) 		# Total number of samples

# print("       Duration : %0.4fs" % D)
# print("  Sample window : %0.4fs" % T)
# print("        Samples : %d" % N)
# print("   Hz detection : <= %0.2f Hz" % (0.5 / T))
# print("    Hz interval : %0.4fs" % ((0.5 / T) / (N // 2 - 1)))

# # Generate the x-coordinates for which to calculate the signal
# x = np.linspace(0, D, N) * 2 * np.pi
# # Calculate the signal
# y = 1 * np.sin(x) + 0.8 * np.sin(4 * x) + 0.5 * np.sin(6 * x)

# # Calculate the intensity and phase of sinusoids present in the signal
# yf = scipy.fftpack.fft(y)
# # Generate the x-axis to match the frequencies to the intensities
# xf = np.linspace(0.0, 0.5 / T, N // 2)

# plt.clf()
# plt.subplot(1, 2, 1)
# plt.plot(y)
# plt.subplot(1, 2, 2)
# plt.plot(xf, 2.0 / N * np.abs(yf[0:(N//2)]))
# plt.grid()
# plt.show()
# ##### End of Example #####

samples = 1000 # 20 seconds, 50 samples per second
offset = 100  # 2 seconds
sample_interval = 2.56 / 128 # 2.56s/window and 128 samples/window gives an interval of 0.020s/sample
sample_frequency = 1.0 / sample_interval

plt.clf()
iPlot = 1
for activity, signal in activityToSignal.items():
	# Grab the signal # Add offset to remove the apparent noise at the beginning of some signals
	y = signal[offset:offset + samples]
	# Generate the x-values corresponding to the signal
	x = np.linspace(offset * sample_interval, (offset + samples) * sample_interval, samples)

	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 0.5 / sample_interval, samples // 2)

	# Plot the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(x, y, label=activity)
	plt.ylim(-1, 1)
	plt.legend()
	iPlot += 1

	# Plot the Fourier transform of the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(xf, 2.0 / samples * np.abs(yf[0:(samples//2)]), label=activity + "(FFT)")
	plt.ylim(0, 0.3)
	plt.legend()
	iPlot += 1
plt.tight_layout() 
plt.suptitle("Signal")
plt.show()

### 4.7
# Create lowpass butterworth filter at 3Hz https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
# Second argument : By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency.
# Nyquist frequency is half the sample_frequency -> 3Hz / (0.5*sample_frequency)
num, denom = butter(5, 3.0 / (0.5 * sample_frequency), btype='lowpass')#, fs=1 / sample_interval, output='sos') # output='sos' ??

plt.clf()
iPlot = 1
for activity, signal in activityToSignal.items():
	# Grab the signal # Add offset to remove the apparent noise at the beginning of some signals
	y = signal[offset:offset + samples]
	y = lfilter(num, denom, y)
	# Generate the x-values corresponding to the signal
	x = np.linspace(offset * sample_interval, (offset + samples) * sample_interval, samples)

	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 0.5 / sample_interval, samples // 2)

	# Plot the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(x, y, label=activity)
	plt.ylim(-1, 1)
	plt.legend()
	iPlot += 1

	# Plot the Fourier transform of the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(xf, 2.0 / samples * np.abs(yf[0:(samples//2)]), label=activity + "(FFT)")
	plt.ylim(0, 0.3)
	plt.legend()
	iPlot += 1
plt.tight_layout()
plt.suptitle("Lowpass : 3Hz")
plt.show()


### HIGHPASS FILTER
num, denom = butter(5, 0.6 / (0.5 * sample_frequency), btype='highpass')#, fs=1 / sample_interval, output='sos') # output='sos' ??

plt.clf()
iPlot = 1
for activity, signal in activityToSignal.items():
	# Grab the signal # Add offset to remove the apparent noise at the beginning of some signals
	y = signal[offset:offset + samples]
	y = lfilter(num, denom, y)
	# Generate the x-values corresponding to the signal
	x = np.linspace(offset * sample_interval, (offset + samples) * sample_interval, samples)

	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 0.5 / sample_interval, samples // 2)

	# Plot the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(x, y, label=activity)
	plt.ylim(-1, 1)
	plt.legend()
	iPlot += 1

	# Plot the Fourier transform of the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(xf, 2.0 / samples * np.abs(yf[0:(samples//2)]), label=activity + "(FFT)")
	plt.ylim(0, 0.3)
	plt.legend()
	iPlot += 1
plt.tight_layout() 
plt.suptitle("Highpass : 0.6Hz")
plt.show()


### BANDPASS FILTER
num, denom = butter(5, [0.6 / (0.5 * sample_frequency), 3.0 / (0.5 * sample_frequency)], btype='band')#, fs=1 / sample_interval, output='sos') # output='sos' ??

plt.clf()
iPlot = 1
for activity, signal in activityToSignal.items():
	# Grab the signal # Add offset to remove the apparent noise at the beginning of some signals
	y = signal[offset:offset + samples]
	y = lfilter(num, denom, y)
	# Generate the x-values corresponding to the signal
	x = np.linspace(offset * sample_interval, (offset + samples) * sample_interval, samples)

	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 0.5 / sample_interval, samples // 2)

	# Plot the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(x, y, label=activity)
	plt.ylim(-1, 1)
	plt.legend()
	iPlot += 1

	# Plot the Fourier transform of the signal
	plt.subplot(6, 2, iPlot)
	plt.plot(xf, 2.0 / samples * np.abs(yf[0:(samples//2)]), label=activity + "(FFT)")
	plt.ylim(0, 0.3)
	plt.legend()
	iPlot += 1
plt.tight_layout() 
plt.suptitle("Bandpass : 0.6Hz, 3Hz")
plt.show()










