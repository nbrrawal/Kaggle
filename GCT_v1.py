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
