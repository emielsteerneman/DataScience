import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load all files
file_activity_labels = "../UCI HAR Dataset/activity_labels.txt"
file_features = "../UCI HAR Dataset/features.txt"
file_X_train = "../UCI HAR Dataset/train/X_train.txt"
file_Y_train = "../UCI HAR Dataset/train/y_train.txt"
file_X_test = "../UCI HAR Dataset/test/X_test.txt"
file_Y_test = "../UCI HAR Dataset/test/y_test.txt"

activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])
X_train = pd.read_csv(file_X_train, delimiter=" ", header=None, skipinitialspace=True)
Y_train = pd.read_csv(file_Y_train, delimiter=" ", header=None, skipinitialspace=True)
X_test = pd.read_csv(file_X_test, delimiter=" ", header=None, skipinitialspace=True)
Y_test = pd.read_csv(file_Y_test, delimiter=" ", header=None, skipinitialspace=True)

Y_train['label'] = Y_train[0].transform(lambda c : activity_labels['activity'][c-1])
Y_test['label']  = Y_test[0].transform(lambda c : activity_labels['activity'][c-1])

# Add labels to the measurements
X_train['label'] = Y_train['label']
X_test['label'] = Y_test['label']

X = pd.concat([X_train, X_test], ignore_index=True)
Y = pd.concat([Y_train, Y_test], ignore_index=True)

### 4.1 a
print("Training set :", X_train.shape)
print("    Test set :", X_test.shape)
print()
### 4.1 b
iFeature = 480
print(" feature :", features['feature'][-1+iFeature])
print("    mean : %0.3f" % X.head(10)[-1+iFeature].mean())
print("  median : %0.3f" % X.head(10)[-1+iFeature].median())
print("  stddev : %0.3f" % X.head(10)[-1+iFeature].std())
print()

### 4.2 a
print("Training labels :", Y_train.shape)
print("    Test labels :", Y_test.shape)
print()
### 4.2 b
bar_heights = Y[0].value_counts(normalize=True).sort_index()
plt.clf() # Clear the current plot
plt.bar(bar_heights.index.values, bar_heights, tick_label=activity_labels['activity']) # Barchart of the distribution of classes
plt.tight_layout()
plt.savefig("4_2b.png", dpi=300) # Save the figure

### 4.3
groupsByLabel = X_train.groupby('label') # Group samples by class
for i in range(0, 10):
	feature = i * 20 + 1 # Random feature
	plt.clf() # Clear the current plot
	groupsByLabel[feature].plot.kde() # get the features for each class, and create a Kernel Density Plot
	plt.title(features[features['id'] == feature]['feature'].tolist()[0]) # Set title to name of feature
	plt.legend(groupsByLabel.groups.keys()) # Add activity labels to legend
	plt.savefig("4_3/4_3_%d.png" % feature, dpi=300) # Save the figure to the folder ./4_3










