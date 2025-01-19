import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from DataTransformation import PrincipalComponentAnalysis
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


df = pd.read_pickle("../../data/interim/removed_outliers_chauvenet.pkl")

df.info()

columns = list(df.columns[:6])

for col in columns:
    df[col] = df[col].interpolate()


df.info()

df[df["label"] == "row"][df["category"] == "heavy"].reset_index()["acc_y"].plot()

setDuration = df[df['set'] == 2].index[-1] - df[df['set'] == 2].index[0]

setDuration.seconds

#avg set duration for each label

for set in df['set'].unique():
    start_time = df[df['set'] == set].index[0]
    stop_time = df[df['set'] == set].index[-1]
    duration = stop_time - start_time
    df.loc[(df['set'] == set),'SetDuration'] = duration.seconds



duration_df = df.groupby('category')['SetDuration'].mean()

lowpass_df = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.28
lowpass_df = LowPass.low_pass_filter(lowpass_df,"acc_y",fs,cutoff)
subset = lowpass_df[df['set'] == 45]
fig, ax = plt.subplots(figsize = (20,10),nrows = 2)

ax[0].plot(subset['acc_y'].reset_index(drop = True))
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop = True))

columns = list(df.columns[:6])
# low pass filetering on all the columns to reduce noise in the data

for col in columns:
    lowpass_df = LowPass.low_pass_filter(lowpass_df,col,fs,cutoff)
    lowpass_df[col] = lowpass_df[col + '_lowpass']
    del lowpass_df[col + '_lowpass']


#pca analysis

pca_df = lowpass_df.copy()
PCA = PrincipalComponentAnalysis()

#determining loading values for each feature
loading_data = PCA.determine_pc_explained_variance(pca_df,columns)



plt.plot(loading_data)
#so by elboy method we can find top 3 features to consider ie 0 1 2

pca_df = PCA.apply_pca(pca_df,columns,3)


pca_df[pca_df['set'] == 35][['pca_1','pca_2','pca_3']].reset_index(drop = True).plot(subplots = True)



df_squares = pca_df.copy()

acc_r = pca_df['acc_x']**2 + pca_df['acc_y']**2 + pca_df['acc_z']**2
gyro_r = pca_df['gyro_x']**2 + pca_df['gyro_y']**2 + pca_df['gyro_z']**2


df_squares['acc_r'] = np.sqrt(acc_r)
df_squares['gyro_r'] = np.sqrt(gyro_r)


from TemporalAbstraction import NumericalAbstraction


df_abs = df_squares.copy()


features = columns + ['acc_r','gyro_r']

NumAbs = NumericalAbstraction()

subsets = []

wd = int(1000/200)

for s in df_abs['set'].unique():
    subset = df_abs[df_abs['set'] == s].copy()
    subset = NumAbs.abstract_numerical(subset,features,wd,"mean")
    subset = NumAbs.abstract_numerical(subset,features,wd,"std")
    subsets.append(subset)
    
df_abs = pd.concat(subsets)

df_abs.info()




df_freq = df_abs.copy().reset_index()

FourAbs = FourierTransformation()

ws = int(1000/200)
fs = int(2800/200)
df_freq = FourAbs.abstract_frequency(data_table=df_freq,cols=["acc_y"],window_size=ws,sampling_rate=fs)


df_cluster = df_freq.copy()

cluster_columns = ['acc_x','acc_y','acc_z']

inertia_values = []
cluster_range = np.arange(3,10)
for k in cluster_range:
    kmeans = KMeans(n_init=20,n_clusters=k,random_state = 42)
    data = df_cluster[cluster_columns]
    y_pred = kmeans.fit(data)
    inertia_values.append(y_pred.inertia_)



 

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal Clusters', fontsize=14)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(cluster_range)  # Ensure all cluster numbers are visible
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#from elbow method we can see that 5 is the optimal cluster number

kmeans = KMeans(n_init = 20,random_state=42,n_clusters=5)

y_pred = kmeans.fit_predict(data)
y_pred.shape

df_cluster['clusters'] = y_pred

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster['clusters'].unique():
    subset = df_cluster[df_cluster['clusters'] == c]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label = c)

ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()



#3d plot for labels
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection = "3d")
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == l]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label = l)

ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()





df_cluster.to_pickle("../../data/interim/processed_data.pkl")

































