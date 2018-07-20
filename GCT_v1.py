# this is sample python code
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import os
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#set plot size
plt.rcParams['figure.figsize']=(20,10)
print(os.listdir("../input/"))
dt = pd.read_csv("e:/dev/Kaggle/input/commodity_trade_statistics_data.csv", na_values=["No Quantity",0.0,''],sep=',')
# Any results you write to the current directory are saved as output.
#first things first, find total number of rows and columns
print("Shape................ :\n" + str(dt.shape))
#List all columns after data is laoded
#temp.columns.values # so we have 8,225,871 rows × 10 columns present in the data
#see names and data types
print("Data type Information.............. :\n" +  str(dt.dtypes))
#sort the data by year to tidy things before moving on
dt = dt.sort_values("year")
#delete temp dataset
#del(temp)
# Preview the data
dt.head(10)
# see missing values
dt.count()
#Preview preliminary statistics as is from the dat awithout cleanup
dt.describe(include="all")
dt = dt.dropna(how='any').reset_index(drop=True)
dt.describe(include="all")
#all missing values 'NaN' are messing up with prelim statisitics, lets treat missing cases by each variable
#dt.describe(include="all")
#First lets filter the data where there is no missing values
#pick first variable
#find unique values present other than nan
print('Country or Area')
dt['country_or_area'].nunique()
dt['country_or_area'].unique()
#there are 209 overall country or areas present in the data, Now lets see frequencies of each of these catergories.. only top 15, since there are a lot of categories
dt['country_or_area'].value_counts()[:15]
