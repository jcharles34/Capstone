import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.ensemble as ske
import matplotlib.pyplot as plt
#%%
stores = pd.read_csv('/Users/johnschweitzer/Downloads/retail-data-analytics/stores data.csv')
sales = pd.read_csv('/Users/johnschweitzer/Downloads/retail-data-analytics/sales data.csv')
features = pd.read_csv('/Users/johnschweitzer/Downloads/retail-data-analytics/Features data.csv')
#%%
pd.to_datetime(sales['Date'], dayfirst=True)
pd.to_datetime(features['Date'], dayfirst=True)
sfdf = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'])
print(sfdf.columns.values)
m = pd.merge(sfdf, stores, on=['Store'])
print(m.columns.values)
m = m.fillna(0)
m1 = m.drop('Date', axis=1)
m1['Type'] = m1['Type'].astype('category')
m1['Type_number'] = m1['Type'].cat.codes
m1 = m1.drop('Type', axis = 1)
m1['Type_number'].unique()
#%%
reg = ske.RandomForestRegressor()
X = m1.drop('Weekly_Sales', axis=1)
Y = m1['Weekly_Sales']
reg.fit(X,Y)
print(reg.feature_importances_)
