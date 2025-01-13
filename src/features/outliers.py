import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import matplotlib as mpl
import math
import scipy

df = pd.read_pickle(r"../../data/interim/resampled_data.pkl")

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 100

columns = list(df.columns[:6])


# box plot for outliers in accelerometer data
df[columns[:3] + ["label"]].boxplot(by="label", layout=(1, 3))
plt.show()


# box plot for outliers in gyroscope data
df[columns[3:6] + ["label"]].boxplot(by="label", layout=(1, 3))
plt.show()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


outlier_col = list(df.columns[:6])

for col in outlier_col:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


# checking for normal distribution

df[outlier_col[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3, 2))
df[outlier_col[3:6] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3, 2))


# As we can see the data is almost normal distributed


# function to mark outliers using the chauvenet's criterion
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


for col in outlier_col:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )

# marking outliers using LocalOutlierFactor
def mark_outliers_loc(dataset,col,k = 20):
    dataset = dataset.copy()
    loc = LocalOutlierFactor(n_neighbors=k)
    data = dataset[col]
    outliers = loc.fit_predict(data)
    X_scores = loc.negative_outlier_factor_
    dataset["is_outlier"] = outliers == -1
    return dataset,X_scores


dataset,X_scores = mark_outliers_loc(df,df.columns[:6])

for col in outlier_col:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="is_outlier", reset_index=True
    )
    
    
# plotting outliers by labels
label = 'squat'
for col in outlier_col:
    dataset = mark_outliers_chauvenet(df[df['label'] == label], col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
    
dataset,X_scores = mark_outliers_loc(df[df['label'] == label],df.columns[:6])

for col in outlier_col:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="is_outlier", reset_index=True
    )    



# choosing chauvenet method and removing the outliers

outliers_removed = df.copy()

for col in outlier_col:
    for label in df['label'].unique():
        #mark Na to the values that are outliers
        dataset = mark_outliers_chauvenet(df[df['label'] == label], col)
        dataset.loc[dataset[col + "_outlier"] == True, col] = np.nan
        outliers_removed.loc[outliers_removed['label'] == label,col] = dataset[col]
        removed = len(dataset) - len(dataset.dropna())
        print(f"Removed {removed} outliers in {col} for label {label}")
        



outliers_removed.info()


outliers_removed.to_pickle('../../data/interim/removed_outliers_chauvenet.pkl')

