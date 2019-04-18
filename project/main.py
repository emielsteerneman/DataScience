import math
import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./surgical_case_durations.csv", delimiter=";", encoding="ISO-8859-1")
dataOG = data.copy() # Original data might be needed for data analysis part





######################################################################
#################### PREPROCESSING ###################################
######################################################################

### Filter out operations that have been done less than 30 times
operationGroups = data.groupby('Operatietype')	# Group data by operation type
operationGroups = operationGroups.size()[30 <= operationGroups.size()] # Only keep operations that have been done at least 30 times
operationTypes = operationGroups.keys().values	# Get types of operations
data = data[data['Operatietype'].isin(operationTypes)] # Filter out operations that are done less than 30 times
# Filter out operations that do not have "Operatieduur"
data = data[data['Operatieduur'].notnull()]

# Replace all commas with dots so floats can be parsed correctly
data.replace(to_replace=",", value=".", inplace=True, regex=True)
# Replace all "Onbekend" occurrences with NaN
data.replace(to_replace="Onbekend", value=np.nan, inplace=True)
# Replace "Ander specialisme" with -1 so that the "chirurg" column can be interpreted nominally
# data['Chirurg'].replace(to_replace="Ander specialisme", value=-1, inplace=True)

# Add the difference in estimated duration and actual duration to the dataset
data['diff'] = data.apply(lambda row : row['Operatieduur'] - row['Geplande operatieduur'], axis=1)





######################################################################
#################### HELPER FUNCTIONS ################################
######################################################################

### Calculate fraction of missing values in columns
def calcFracMissingPerCol(df):
	fracMissingPerCol = {}

	columnNames = list(df.columns.values)
	for i, c in enumerate(columnNames):
		col = data[c]
		total = len(col)
		nan = len(col[col.isna()])
		fracMissingPerCol[c] = 1 - (nan / total)
		# print(i, "\t%0.2f " % (1 - nan / total), c)
	return fracMissingPerCol

def calcBestSplit(df, cols, target, log=True, ignoreNaN=False):
	results = []

	for col in cols:
		groups = df.fillna(-1).groupby(col)	# Group the data by the column
		# groups = df.groupby(col)	# Group the data by the column
		keys = groups.groups.keys() 		# Get the different types in the groups
		if ignoreNaN:
			keys = keys - [-1]
		varTotal = 0						# Variance accumulator
		fracMissingPerCol = calcFracMissingPerCol(df)

		nTarget = len(df[target])		# Number of datapoints in target
		varTarget = df[target].var()	# Variance of target

		if len(keys) < 2:			# If the group cannot be split
			continue				#     Skip the group

		for key in keys:			# For each type
			group = groups.get_group(key)		# Get the group
			var = group[target].var()			# Calculate its variance of the target column
			if math.isnan(var):					# If the variance is NaN (happens when group only has 1 value)
				var = 1							#     Set variance to 0
			weight = len(group) / nTarget 		# Calculate weight of variance
			varTotal += var * weight 			# Add weighted variance to total variance of type
			# print("    ", col, "|", key, "var=%0.4f" % var, "weight=%0.4f" % weight, "fm=%0.4f" % fracMissingPerCol[col], "n=%d" % len(group))

		results.append([col, len(keys), varTotal, varTotal / varTarget])

	if len(results) == 0:
		if log:
			print("=" * 30, "calcBestSplit", "=" * 30)
			print("No results".rjust(35))
		return []

	results.sort(key=lambda r : r[2])
	
	if log:
		print("=" * 30, "calcBestSplit", "=" * 30)
		print("CATEGORY".rjust(35), "# TYPES".rjust(10), "VARIANCE".rjust(10))
		# print("TARGET".rjust(35), "-".rjust(10), "1.00".rjust(10))
		for col, nKeys, var, frac in results[:3]:
			print(col.rjust(35), ("%s" % nKeys).rjust(10), ("%0.3f" % frac).rjust(10))
		# for col, nKeys, var, frac in results:
		# 	print(col.rjust(35), ("& %s" % nKeys).rjust(10), ("& %0.3f" % frac).rjust(10), "\\\\")	


	return results[0]

def calcBestSplitNum(_df, cols, target, log=True):
	resultsTotal = []

	for col in cols:
		df = _df[[col, target]].copy()
		df[col] = df[col].fillna(df[col][df[col].notnull()].mean())

		values = df[col].values
		valuesUnique = np.unique(values)
		nTarget = len(values)
		varTarget = df[target].var()

		thresholds = []
		for i in range(len(valuesUnique)-1):
			thresholds.append((valuesUnique[i]+valuesUnique[i+1])/2)

		results = []

		for thresh in thresholds:
			myFilter = df[col] < thresh
			left  = df[target][myFilter]
			right = df[target][~myFilter]

			nLeft = len(left)
			nRight = len(right)

			varLeft  = left.values.var()
			varRight = right.values.var()

			fracLeft  = nLeft  / nTarget
			fracRight = nRight / nTarget

			varTotal = varLeft * fracLeft + varRight * fracRight
			# print("tresh", thresh, "left ", len(left), "right", len(right), "     ", varTotal, "    ", col)
			results.append([thresh, varTarget, varTotal, varLeft, fracLeft, varRight, fracRight, col])

		
		results.sort(key=lambda r : r[2])
		if log:
			print("\n", "THRESH".rjust(50), " VARtarget", "  VARtotal", "   VARleft", "  FRACleft", "  VARright", " FRACright")
			for t, vTar, vTot, vL, fL, vR, fR, _ in results[:1]:
				print("%s"%col[:38].rjust(40), 
					("%0.2f"%t).rjust(10),
					("%0.2f"%vTar).rjust(10),
					("%0.2f"%vTot).rjust(10),
					("%0.2f"%vL).rjust(10),
					("%0.2f"%fL).rjust(10),
					("%0.2f"%vR).rjust(10),
					("%0.2f"%fR).rjust(10))
		if 0 < len(results):
			resultsTotal.append(results[0])
	
	resultsTotal.sort(key=lambda r : r[2])
	return resultsTotal[0]

def applyLinearRegression(df, source, target):
	output = df[target].copy()
	output = output.fillna(output[output.notnull()].mean())

	model = Sequential()
	model.add(Dense(1, input_shape=(1,), activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam')

	inputCol = df[source].copy()
	inputCol = inputCol.fillna(inputCol[inputCol.notnull()].mean())
	hist = model.fit(inputCol, output, batch_size=1, epochs=20, verbose=0)
	print(source.rjust(30), hist.history['loss'][-1])

def applyNN(_df, columns, target):
	df = _df.copy()

	output = df[target].copy()
	output = output.fillna(output[output.notnull()].mean())

	for col in columns:
		df[col] = df[col].fillna(df[col][df[col].notnull()].mean())

	model = Sequential()
	model.add(Dense(10, input_shape=(len(columns),), activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense( 1, activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam')

	early_stopping_monitor = EarlyStopping(monitor='loss', patience=20, verbose=0)
	hist = model.fit(df[columns], output, batch_size=100, epochs=5000, verbose=0, callbacks=[early_stopping_monitor])

	return model, hist


### Check which columns are numerical or categorical
columnNames = list(data.columns.values)
numericalCols = []
categoricalCols = []
for col in columnNames:
	try:
		data[col] = data[col].apply(lambda x : float(x))
		numericalCols.append(col)
	except Exception as e:
		# print("Non-numerical:", col, e)
		categoricalCols.append(col)

### Remove numerical columns that should not be used for predicting the Operatieduur
numericalCols = list(set(numericalCols) - set(["Operatieduur", "Geplande operatieduur", "Ziekenhuis ligduur", "IC ligduur", "diff"]))





######################################################################
#################### DATA ANALYSIS ###################################
######################################################################

### Print some basic information about the dataset
if False:
	diffs = data[data['diff'].notnull()]['diff']
	print()
	print(" overestimated : %d  \t %d minutes" % (diffs[diffs > 0].size, sum(diffs[diffs > 0])))
	print("     estimated : %d  \t %d minutes" % (diffs[diffs== 0].size, sum(diffs[diffs== 0])))
	print("underestimated : %d  \t %d minutes" % (diffs[diffs < 0].size, sum(diffs[diffs < 0])))

	groups = dataOG.groupby('Operatietype')		# Group data by operation type
	operationTypesBefore = groups.groups.keys()	# Get all types of operations
	nOperationsBefore = len(dataOG)				# Count the total number of operations
	
	groups = data.groupby('Operatietype')		# Group data by operation type
	operationTypesAfter = groups.groups.keys()	# Get all types of operations
	nOperationsAfter = len(data)				# Count the total number of operations
	
	print()	
	print("Number of operation types              : %d   (%d operations)" % (len(operationTypesBefore), nOperationsBefore))
	print("Number of operation types after filter :  %d   (%d operations)" % (len(operationTypesAfter), nOperationsAfter))

### Build a graph to justify the threshold of 30 operations
if False:
	groups = dataOG.groupby("Operatietype")
	gI = []
	gTypes = []
	gOperations = []
	for i in range(0, 500, 5):
		myFilter = i <= groups.size()		# Create filter
		types = groups.size()[myFilter]		# Filter groups 
		nTypes = len(types)					# Count the number of groups left
		nOperations = sum(types.values)		# Count the number of operations that the groups left hold
		gI.append(i)						# Store i
		gTypes.append(nTypes)				# Store number of types
		gOperations.append(nOperations)		# Store number of operations
	
	plt.clf()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(gI, gTypes, c="red")
	ax1.tick_params('y', colors='red')
	ax1.axvline(x=20, c="black")
	ax1.set_ylabel("# distinct operation types", color="red")

	ax2.plot(gI, gOperations, c="blue")
	ax2.tick_params('y', colors='blue')
	ax2.set_ylabel("# operations", color="blue")

	ax1.set_xlabel("Threshold value")
	fig.tight_layout()
	plt.show()

### Plot the estimated duration against the error
if False:
	groups = data.groupby("Operatietype")
	commonOperations = groups.groups.keys()
	plt.clf()
	for t in commonOperations:
		group = groups.get_group(t)
		group = group[group['Geplande operatieduur'] <= 400]
		x = group['Geplande operatieduur']
		y = group['diff']
		plt.scatter(x, y, s=3)

	plt.title("Planned duration plotted against the difference (or error)")
	plt.xlabel("Planned duration")
	plt.ylabel("Difference between planned and actual duration")
	plt.grid()
	plt.show()

### Check if duration of common operations are consistent
if False:
	# print("\n### Check if duration of common operations are consistent")
	groups = data.groupby("Operatietype")
	commonOperations = groups.groups.keys()
	plt.clf()
	plt.xticks(range(len(commonOperations)), commonOperations, rotation='vertical')
	for t in commonOperations:
		planned = groups.get_group(t)['Geplande operatieduur'].values
		durations = groups.get_group(t)['Operatieduur'].values
		# print("%s : planned mean=%7.3f    mean=%7.3f    median=%7.3f    var=%8.3f    n=%d" % (t.rjust(40), planned.mean(), durations.mean(), np.median(durations), durations.var(), len(durations)))
		plt.scatter(len(durations) * [t], y=durations, s=3)
	plt.ylabel("Duration in minutes")
	plt.title("Operation duration of operations that have occured over 30 times")
	plt.grid()
	plt.tight_layout()
	plt.show()

### Check if certain operations are consistenty underestimated or overestimated
if False:
	# print("\n### Check if certain operations are consistenty underestimated or overestimated")
	groups = data.groupby("Operatietype")
	commonOperations = groups.groups.keys()
	plt.clf()
	plt.xticks(range(len(commonOperations)), commonOperations, rotation='vertical')
	for t in commonOperations:
		errors = groups.get_group(t)['diff'].values
		# print("%s : mean=%7.3f    median=%7.3f    var=%8.3f    n=%d" % (t.rjust(40), errors.mean(), np.median(errors), errors.var(), len(errors)))
		plt.scatter(len(errors) * [t], y=errors, s=3)
	plt.ylabel("Error in minutes")
	plt.title("Overestimation-error of operations that have occured over 30 times")
	plt.grid()
	plt.tight_layout()
	plt.show()

### Check which operations are underestimated the most
if False:
	print("\n\n############### Operations that are underestimated the most:")
	groups = data.groupby("Operatietype")
	commonOperations = groups.groups.keys()
	dataPerType = []
	for t in commonOperations:
		group = groups.get_group(t)
		underestimated = group[group['diff'] > 10]
		minutes = sum(underestimated['diff'].values)
		nUnderestimated = len(underestimated)
		nTotal = len(group)
		percentage = nUnderestimated / nTotal
		dataPerType.append([t, percentage, nTotal, nUnderestimated, minutes])

	dataPerType.sort(key = lambda x : x[1], reverse=True)
	print("TYPE".rjust(40), "  %   ", "X OUT OF N".rjust(14), "  MINUTES")
	for t, p, nT, nU, m in dataPerType:
		print(t[:38].rjust(40), "  %0.2f" % p, ("%d"%nU).rjust(7), "/", ("%d"%nT).rjust(4), ("%d"%m).rjust(9))

### Check if predictions would have been more accurate if the mean would've been used
if False:
	print("\n\n############### Check if predictions would have been more accurate if the mean would've been used")
	groups = data.groupby("Operatietype")
	commonOperations = groups.groups.keys()
	totalPlannedError = 0
	totalMeanError = 0
	dataPerType = []
	for t in commonOperations:
		plannedError = sum(abs(groups.get_group(t)['diff'].values))
		if(math.isnan(plannedError)): # There seems to be a faulty value somewhere in "CABG + Pacemakerdraad tijdelijk"
			continue

		durations = groups.get_group(t)['Operatieduur'].values
		durationsMean = durations.mean()
		meanError = sum(abs(durations - durationsMean))
		if(meanError == 0):
			meanError = 1

		totalPlannedError += plannedError
		totalMeanError += meanError
		dataPerType.append([t, plannedError, meanError, plannedError - meanError, plannedError/meanError])

	dataPerType.sort(key = lambda x : x[4], reverse=True)
	print("TYPE".rjust(40), "  PLANNED ERROR    MEAN ERROR    DIFFERENCE     P/M")
	for t, p, m, d, f in dataPerType:	
		print("%s       %9.0f     %9.0f     %9.2f    %0.2f" % (t[:38].rjust(40), p, m, d, f ))
	print("SUMMARY".rjust(40), "      %9.0f     %9.0f     %9.2f    %0.2f" % (totalPlannedError, totalMeanError, totalPlannedError - totalMeanError, totalPlannedError/totalMeanError ))

### Compare correlation of all numerical columns against the operation duration
if False:
	print("COLUMN".rjust(40), "CORRELATION")
	print("\n############### Compare correlation of all numerical columns against the operation duration")
	for col in numericalCols:
		c = data[col].corr(data["Operatieduur"])
		print(col[:38].rjust(40), "    %0.3f" % c)





######################################################################
#################### MODEL ###########################################
######################################################################

result = calcBestSplit(data, categoricalCols, "Operatieduur")

### Store intermediate values to figure out which column influences operation type the most
intermediate = {}
groups = data.groupby([result[0]])
keys = groups.groups.keys()
### Fill intermediate values with placeholder
for key in keys:
	intermediate[key] = {"col" : "none", "fV" : 10}



### Try splitting again, after splitting once 
if False:
	groups = data.groupby([result[0]])
	keys = groups.groups.keys()
	n = len(data)

	print("OPERATION".rjust(40), "CATEGORY".rjust(30), "# TYPES".rjust(10), "VARIANCE".rjust(10))
	for key in keys:
		group = groups.get_group(key)
		col, nT, v, fV = calcBestSplit(group, categoricalCols, "Operatieduur", log=False, ignoreNaN=False)
		print("%s" % key[:38].rjust(40), col[:28].rjust(30), ("%s" % nT).rjust(10), ("%0.3f" % fV).rjust(10))
		# print("TARGET".rjust(35), "-".rjust(10), "1.00".rjust(10))
		# for col, nKeys, var, frac, n in results[:3]:
		# 	print(col.rjust(35), ("%s" % nKeys).rjust(10), ("%0.3f" % frac).rjust(10))
		if fV < intermediate[key]["fV"]:
			intermediate[col] = {"col" : col, "fV" : fV}

### Try splitting on numerical, after splitting once 
if False:
	groups = data.groupby([result[0]])
	keys = groups.groups.keys()
	n = len(data)

	for key in keys:
		group = groups.get_group(key)
		# print("\n", key, "(%d samples)" % len(group))
		t, vTar, vTot, vL, fL, vR, fR, col = calcBestSplitNum(group, numericalCols, "Operatieduur", log=False)
		fV = vTot / vTar
		print("%s" % key[:38].rjust(40), col[:28].rjust(30), ("%0.3f" % t).rjust(10), ("%0.3f" % fV).rjust(10))

		if fV < intermediate[key]["fV"]:
			intermediate[key] = {"col" : col, "fV" : fV}

### Apply Linear Regression to numerical columns
if False:
	for col in numericalCols:
		applyLinearRegression(data, col, 'Operatieduur')

### Check if the correlation of any of the numerical attributes is better when split on Operatietype
if False:
	groups = data.groupby([result[0]])
	keys = groups.groups.keys()
	n = len(data)


	for key in keys:
	
		weightedCorrs = {}
		for col in numericalCols:
			weightedCorrs[col] = 0

		group = groups.get_group(key)
		frac = len(group) / n
		for col in numericalCols:
			corr = group[col].corr(group["Operatieduur"])
			if math.isnan(corr):
				corr = 0
			weightedCorrs[col] += corr# * frac
			# print(key, "|", col, "|", n, len(group), corr, weightedCorrs[col])

		print(key)
		for col in numericalCols:
			print("    %0.3f" % weightedCorrs[col], col)

### Run neural network on all numerical cols after grouping by operation type
if False:
	print("\nApplying neural network on numerical columns after grouping by operation type")
	df = data.copy()
	df["Operatieduur"] = df["Operatieduur"].apply(lambda x: x / 100)
	df["Geplande operatieduur"] = df["Geplande operatieduur"].apply(lambda x: x / 100)

	for col in numericalCols:
		df[col] = df[col].fillna(df[col][df[col].notnull()].mean())

	def testModel(model, df):
		actual = df["Operatieduur"].values
		planned = df["Geplande operatieduur"].values
		predicted = model.predict(df[numericalCols])[:,0]

		errorPlanned   = sum(abs(actual-planned))
		errorPredicted = sum(abs(actual-predicted))
		errorMean = sum(abs(actual - actual.mean()))
		# print(actual)
		# print(planned)
		# print(predicted)
		# print("  planned:", errorPlanned)
		# print("predicted:", errorPredicted)
		return errorPlanned, errorPredicted, errorMean

	mask = np.random.rand(len(df)) < 0.8
	train = df[mask]
	test = df[~mask]
	model, hist = applyNN(train, numericalCols, "Operatieduur")
	errorPlanned, errorPredicted, errorMean = testModel(model, test)

	print("OPERATION".rjust(40), "PLANNED ERROR".rjust(16), "MEAN ERROR".rjust(16), "MODEL ERROR".rjust(16), )
	print("Original".rjust(40),
			("  %0.2f" % errorPlanned).rjust(16), 
			("& %0.2f" % errorMean).rjust(16),
			("& %0.2f" % errorPredicted).rjust(16))

	groups = df.groupby([result[0]])
	for key in groups.groups.keys():
		group = groups.get_group(key)
		mask = np.random.rand(len(group)) < 0.8
		train = group[mask]
		test = group[~mask]
		model, hist = applyNN(train, numericalCols, "Operatieduur")
		errorPlanned, errorPredicted, errorMean = testModel(model, test)
		print(key[:38].rjust(40), 
			("  %0.2f" % errorPlanned).rjust(16), 
			("& %0.2f" % errorMean).rjust(16),
			("& %0.2f" % errorPredicted).rjust(16))



### Print intermediate values
print("\n\n")
for key in keys:
	print("%s" % key[:38].rjust(40), "&", intermediate[key]["col"][:28].rjust(30), "&", ("%0.3f" % intermediate[key]["fV"]).rjust(10))








