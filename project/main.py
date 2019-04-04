import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./surgical_case_durations.csv", delimiter=";", encoding="ISO-8859-1")
columnNames = list(data.columns.values)
print(columnNames)

# for c in columnNames:
# 	print("\n")
# 	print(data[c][1:5])

data['diff'] = data.apply(lambda row : row['Operatieduur'] - row['Geplande operatieduur'], axis=1)
diffs = data[data['diff'].notnull()]['diff']
print()
print(" overestimated : %d" % diffs[diffs > 0].size)
print("     estimated : %d" % diffs[diffs == 0].size)
print("underestimated : %d" % diffs[diffs < 0].size)


groups = data.groupby('Operatietype')
operationTypes = groups.groups.keys()
print("Number of operation types     : %d" % len(operationTypes))
myFilter = groups.size() == 1
print("Number of operations filtered : %d" % len(groups.size()[myFilter]))
onces = list(groups.size()[myFilter].keys())
rows = data[data['Operatietype'].isin(onces)][['Operatietype', 'Geplande operatieduur', 'Operatieduur', 'diff']]

# for once in onces:
# 	print(once)

print(rows.to_string())



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




