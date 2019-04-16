import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./surgical_case_durations.csv", delimiter=";", encoding="ISO-8859-1")

# print((data))
data.replace(to_replace="Onbekend", value=np.nan, inplace=True)
# print(data.isna().sum().values.sum())
# exit()

# wer = data[data['Operatietype'] == "Bentall procedure"]['Operatieduur']
# print(wer)
# print(wer.var())
# exit()


### Filter out operations that have been done less than 30 times
operationGroups = data.groupby('Operatietype')	# Group data by operation type
operationGroups = operationGroups.size()[30 <= operationGroups.size()] # Only keep operations that have been done at least 30 times
operationTypes = operationGroups.keys().values	# Get types of operations
data = data[data['Operatietype'].isin(operationTypes)] # Filter out operations that are done less than 30 times

# Replace all commas with dots so floats can be parsed correctly
data.replace(to_replace=",", value=".", inplace=True, regex=True)
# Replace all "Onbekend" occurrences with NaN
data.replace(to_replace="Onbekend", value=np.nan, inplace=True)
# Replace "Ander specialisme" with -1 so that the "chirurg" column can be interpreted nominally
# data['Chirurg'].replace(to_replace="Ander specialisme", value=-1, inplace=True)

# Add the difference in estimated duration and actual duration to the dataset
data['diff'] = data.apply(lambda row : row['Operatieduur'] - row['Geplande operatieduur'], axis=1)


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

def calcBestSplit(df, cols, target, log=True):
	results = []

	for col in cols:
		groups = df.fillna(-1).groupby(col)	# Group the data by the column
		keys = groups.groups.keys() 		# Get the different types in the groups
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



	if log:
		results.sort(key=lambda r : r[2])
		print("\n", "=" * 30, "calcBestSplit", "=" * 30)
		print("CATEGORY".rjust(35), "# TYPES".rjust(10), "VARIANCE".rjust(10))
		print("TARGET".rjust(35), "-".rjust(10), "1.00".rjust(10))
		for col, nKeys, var, frac in results:
			print(col.rjust(35), ("%s" % nKeys).rjust(10), ("%0.2f" % frac).rjust(10))

	return results[0]



# nanFracs = [0] * len(columnNames)
# def x(row):
# 	nanFracs[len(row[row.isna()])] += 1
# data.apply(x , axis=1)
# print(list(enumerate(nanFracs)))


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
print("Numerical columns (%d):" % len(numericalCols))
print("    ", ", ".join(numericalCols))
# Print categorical columns
print("Categorical columns (%d):" % len(categoricalCols))
print("    ", ", ".join(categoricalCols))

### Compare correlation of all numerical columns against the operation duration
print("\nCompare correlation of all numerical columns against the operation duration")
for col in numericalCols:
	c = data[col].corr(data["Operatieduur"])
	print("    %0.3f" % c, col)


result = calcBestSplit(data, categoricalCols, "Operatieduur")

if True:
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

exit()


diffs = data[data['diff'].notnull()]['diff']
print()
print(" overestimated : %d  \t %d minutes" % (diffs[diffs > 0].size, sum(diffs[diffs > 0])))
print("     estimated : %d  \t %d minutes" % (diffs[diffs== 0].size, sum(diffs[diffs== 0])))
print("underestimated : %d  \t %d minutes" % (diffs[diffs < 0].size, sum(diffs[diffs < 0])))

groups = data.groupby('Operatietype')	# Group data by operation type
operationTypes = groups.groups.keys()	# Get all types of operations
nOperations = len(data)					# Count the total number of operations
myFilter = groups.size() >= 30			# Define a filter
nOperationsAfterFilter = sum(groups.size()[myFilter].values) # Count the total number of operations after applying the filter
print()	
print("Number of operation types              : %d   (%d operations)" % (len(operationTypes), nOperations))
print("Number of operation types after filter :  %d   (%d operations)" % (len(groups.size()[myFilter]), nOperationsAfterFilter))



### Build a graph to justify the threshold of 30 operations
if False:
	plt.clf()
	gI = []
	gTypes = []
	gOperations = []
	for i in range(0, 500):
		myFilter = i <= groups.size()		# Create filter
		types = groups.size()[myFilter]		# Filter groups 
		nTypes = len(types)					# Count the number of groups left
		nOperations = sum(types.values)		# Count the number of operations that the groups left hold
		gI.append(i)						# Store i
		gTypes.append(nTypes)				# Store number of types
		gOperations.append(nOperations)		# Store number of operations
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
	plt.clf()
	commonOperations = groups.size()[30 <= groups.size()].keys()
	for t in commonOperations:
		group = groups.get_group(t)
		group = group[group['Geplande operatieduur'] <= 400]
		x = group['Geplande operatieduur']
		y = group['diff']
		plt.scatter(x, y, s=3)

	plt.title("Planned duration plotted against the difference (or error)")
	plt.xlabel("Planned duration")
	plt.ylabel("Difference between planned and actual duration")
	plt.show()

### Plot the Euroscore2 against the error
if False:
	plt.clf()
	commonOperations = groups.size()[30 <= groups.size()].keys()
	for t in commonOperations:
		group = groups.get_group(t)
		# group = group[group['Geplande operatieduur'] <= 400]
		group = group[['Euroscore2', 'diff']]
		group = group.dropna()
		group['Euroscore2'] = group['Euroscore2'].apply(lambda f : float(f.replace(",", ".")))
		x = group['Euroscore2']
		y = group['diff']
		plt.scatter(x, y, s=3)

	plt.title("Euroscore2 plotted against the difference (or error)")
	plt.xlabel("Euroscore2")
	plt.ylabel("Difference between planned and actual duration")
	plt.show()
	exit()

### Check which operations are underestimated the most
print("\nOperations that are underestimated the most:")
commonOperations = groups.size()[30 <= groups.size()].keys()
underestimationPerType = []
for t in commonOperations:
	group = groups.get_group(t)
	underestimated = group[group['diff'] > 10]
	minutes = sum(underestimated['diff'].values)
	nUnderestimated = len(underestimated)
	nTotal = len(group)
	percentage = nUnderestimated / nTotal
	underestimationPerType.append([t, percentage, nTotal, nUnderestimated, minutes])

underestimationPerType.sort(key = lambda x : x[1], reverse=True)
for t, p, nT, nU, m in underestimationPerType:
	print(t.rjust(70), "%0.2f" % p, ("%d"%nU).rjust(4), "/", ("%d"%nT).rjust(4), "   ", m, "minutes")

# exit()



### Check if duration of common operations are consistent
print("\n### Check if duration of common operations are consistent")
commonOperations = groups.size()[50 < groups.size()].keys()
plt.clf()
plt.xticks(range(len(commonOperations)), commonOperations, rotation='vertical')
for t in commonOperations:
	planned = groups.get_group(t)['Geplande operatieduur'].values
	durations = groups.get_group(t)['Operatieduur'].values
	print("%s : planned mean=%7.3f    mean=%7.3f    median=%7.3f    var=%8.3f    n=%d" % (t.rjust(40), planned.mean(), durations.mean(), np.median(durations), durations.var(), len(durations)))
	plt.scatter(len(durations) * [t], y=durations, s=3)
plt.ylabel("Duration in minutes")
plt.title("Operation duration of operations that have occured over 50 times")
plt.grid()
plt.tight_layout()
# plt.show()

### Check if certain operations are consistenty underestimated or overestimated
print("\n### Check if certain operations are consistenty underestimated or overestimated")
commonOperations = groups.size()[50 < groups.size()].keys()
plt.clf()
plt.xticks(range(len(commonOperations)), commonOperations, rotation='vertical')
for t in commonOperations:
	errors = groups.get_group(t)['diff'].values
	print("%s : mean=%7.3f    median=%7.3f    var=%8.3f    n=%d" % (t.rjust(40), errors.mean(), np.median(errors), errors.var(), len(errors)))
	plt.scatter(len(errors) * [t], y=errors, s=3)
plt.ylabel("Error in minutes")
plt.title("Overestimation-error of operations that have occured over 50 times")
plt.grid()
plt.tight_layout()
# plt.show()


### Check if predictions would have been more accurate if the mean would've been used
print("\n### Check if predictions would have been more accurate if the mean would've been used")
commonOperations = groups.size()[-1 < groups.size()].keys()
totalPlannedError = 0
totalMeanError = 0
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

	print("%s : planned error=%9.3f    mean error=%9.3f    diff=%9.3f    a/b=%0.3f" % (t[:40].rjust(42), plannedError, meanError, plannedError - meanError, plannedError/meanError ))

print("\nSummary : planned error=%0.3f    mean error=%0.3f    diff=%0.3f    a/b=%0.3f" % (totalPlannedError, totalMeanError, totalPlannedError - totalMeanError, totalPlannedError/totalMeanError ))




