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
#Create datatime data types for dates
pd.to_datetime(sales['Date'], dayfirst=True)
pd.to_datetime(features['Date'], dayfirst=True)
#Merge imported dataframes into one dataframe
sfdf = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'])
print(sfdf.columns.values)
m = pd.merge(sfdf, stores, on=['Store'])
print(m.columns.values)
#Make NA data numerical
m = m.fillna(0)
m1 = m.drop('Date', axis=1)
m1['Type'] = m1['Type'].astype('category')
m1['Type_number'] = m1['Type'].cat.codes
m1 = m1.drop('Type', axis = 1)
m1['Type_number'].unique()
#create a column to adjust for store size
m1['Sales/Size'] = m1['Weekly_Sales'] / m1['Size']
#Create a column for total markdown
m1['Total_markdown'] = m1['MarkDown1'] + m1['MarkDown2'] + m1['MarkDown3'] + m1['MarkDown4'] + m1['MarkDown5']
#summary statistics
m1.describe()
#%%
#Create a dataframe that does not have holidays
nh = m1.loc[m1['IsHoliday'] == False]
#Create a dataframe that only includes holidays 
H = m1.loc[m1['IsHoliday']== True]
#obtain summary statistics
print(nh['Weekly_Sales'].describe())
print(H['Weekly_Sales'].describe())
#plot weekly sales on holiday and non holiday weeks
nh.plot(y='Weekly_Sales', kind='hist', bins =100)
H.plot(y='Weekly_Sales', kind ='hist', bins=100)
#%%
#Visualize data
m1.plot(x='Size', y='Weekly_Sales', kind = 'scatter')
m1.plot(y='Sales/Size', kind = 'hist', bins=100)
m1.plot(x='Store', y = 'Weekly_Sales', kind = 'scatter')
m1.plot(x='Store', y = 'Sales/Size', kind = 'scatter')
m1.plot(x='Dept', y = 'Weekly_Sales', kind ='scatter')
m1.plot(x='Temperature', y = 'Weekly_Sales', kind ='scatter')
m1.plot(x='CPI', y='Weekly_Sales', kind ='scatter')
m1.plot(x='CPI', y='Fuel_Price', kind = 'scatter')
m1.plot(x='Type_number', y = 'Sales/Size', kind = 'scatter')
m1.plot(x='Type_number', y = 'Weekly_Sales', kind = 'scatter')
m1.plot(x='Total_markdown', y = 'Weekly_Sales', kind = 'scatter')
#%%
reg = ske.RandomForestRegressor()
#Split data into target and output variables and drop Sales/Size column
X1 = m1.drop('Weekly_Sales', axis=1)
X = X1.drop('Sales/Size', axis=1)
Y = m1['Weekly_Sales']
#Fit RFR to data
reg.fit(X,Y)
print(reg.feature_importances_)
#plot feature importances

#Data Quality Check (negative values)
#Create a dataframe that has markdowns but no holidays
#Create a dataframe that has markdowns and holidays
#Compute summary statistics on both
#Run P test to see if they are significantly different

#Run simple linear regression
#Self-learning models
