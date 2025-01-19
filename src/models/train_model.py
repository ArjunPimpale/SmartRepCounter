import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
df = pd.read_pickle('../../data/interim/processed_data.pkl')


#train and test split
features = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']

encoder = LabelEncoder()

# Fit and transform the 'Label' column
df['Encoded_Label'] = encoder.fit_transform(df['label'])


