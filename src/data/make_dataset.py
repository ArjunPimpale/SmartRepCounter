import pandas as pd
from glob import glob


files = glob(r"..\..\data\raw\MetaMotion\*.csv")

def cat_data(files):
    data_path = "..\\..\\data\\raw\\MetaMotion\\"
 
    acc_count = 1
    gyro_count = 1
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    for f in files:
        candidate = f.split("-")[0].lstrip(data_path)
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        df = pd.read_csv(f)
        
        df['category'] = category
        df['label'] = label
        df['candidate'] = candidate
        
        if "Accelerometer" in f:
            df["set"] = acc_count
            acc_df = pd.concat([acc_df, df])
            acc_count += 1
        else:
            df["set"] = gyro_count
            gyro_df = pd.concat([gyro_df, df])
            gyro_count += 1
            
            
            
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyro_df.index = pd.to_datetime(gyro_df['epoch (ms)'], unit='ms')


    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del gyro_df['epoch (ms)']
    del gyro_df['time (01:00)']
    del gyro_df['elapsed (s)']
    
    return acc_df, gyro_df

acc_df,gyro_df = cat_data(files)

merged_data = pd.concat([acc_df.iloc[:,:3],gyro_df], axis = 1)

merged_data.columns = [
    'acc_x' ,
    'acc_y' ,
    'acc_z' ,
    'gyro_x' ,
    'gyro_y' ,
    'gyro_z' ,
    'category',
    'label',
    'candidate',
    'set'
    
]

sampling = {
    'acc_x' : 'mean',
    'acc_y' : 'mean',
    'acc_z' : 'mean',
    'gyro_x' : 'mean',
    'gyro_y' : 'mean',
    'gyro_z': 'mean',
    'category' :'last',
    'label': 'last',
    'candidate' :'last',
    'set' :'last'    
    
}


data_chunks_by_days = [g for n,g in merged_data.groupby(pd.Grouper(freq='D'))]

data_chunks_by_days[0]

resampled_data = pd.concat([g.resample(rule = '200ms').apply(sampling).dropna() for g in data_chunks_by_days])
#divide merged data into chunks where grouped by one day 
#and calculate the mean of each feature in each chunk
#then save the data to a csv file

resampled_data.info()

resampled_data["set"] = resampled_data["set"].astype("int")


resampled_data.to_pickle(r"..\..\data\interim\resampled_data.pkl")

