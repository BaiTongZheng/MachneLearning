
import pandas as pd 
import numpy as np 
from pandas import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

data_train = pd.read_csv("/home/baitong/Data/all/train.csv")
data_train['Sex_Pclass'] = data_train.Sex + "_" + data_train.Pclass.map(str)

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
