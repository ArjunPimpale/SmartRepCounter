import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from DataTransformation import PrincipalComponentAnalysis



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










