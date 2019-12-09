# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:16:26 2019

@author: Narayan
"""

import os
os.chdir("E:\\Dev\\Kaggle\\ashrae\\")
import pandas as pd 
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
os.listdir()
train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
weather_train= pd.read_csv("weather_train.csv")
weather_test= pd.read_csv("weather_test.csv")
building_meta= pd.read_csv("building_metadata.csv")

#line chart 
from matplotlib import pyplot
series = train[(train.building_id==114) & (train.meter==3)][["timestamp","meter_reading"]]
series.plot()
pyplot.show()
series = train[(train.building_id==114) & (train.meter==0)][["timestamp","meter_reading"]]
series.plot()
pyplot.show()

series.plot(style='k.')
pyplot.show()
#histogram 
series.hist()
pyplot.show()
#kde - line plot  
series.plot(kind='kde')
pyplot.show()
#boxplot 
series.boxplot()
pyplot.show()
import numpy as np 

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
weather_train = reduce_mem_usage(weather_train)
weather_test = reduce_mem_usage(weather_test)
building_meta = reduce_mem_usage(building_meta)
import matplotlib.pyplot as plt 
#overall meter reading 
y_mean_time = train.groupby('timestamp').meter_reading.mean()
y_mean_time.plot(figsize=(20, 8))

y_mean_time.rolling(window=10).std().plot(figsize=(20, 8))
ax = plt.axhline(y=0.009, color='red')
