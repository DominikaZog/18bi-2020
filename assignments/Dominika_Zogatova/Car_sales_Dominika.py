#!/usr/bin/env python
# coding: utf-8 

# # Car sales analysis
#  **Dominika Zogatová**   
#  **BI 2020**  
#    
#  Data source: https://github.com/jbrownlee/Datasets/blob/master/monthly-car-sales.csv

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet # data forcasting
import warnings # ignore warinings

cars = pd.read_excel('monthly_car_sales.xlsx')

cars = pd.DataFrame(cars)

cars.head()


# # Inspecting the dataset  
# 
# There are no missing vaules in the dataset.

# In[7]:


print('\n Number of missing values in each column')
cars.isnull().sum(axis = 0) #missing vaules in every column

# df.isnull().sum(axis = 1) #missing values in every row

# cars.dropna(inplace = True) # drop missing values 


# In[8]:


years = cars['Month'].str.split("-", n = 1, expand = True) # using .str.split() to split every row in a column

cars["Year"] = years[0]

print('\n Summary statistics for each year between 1960 and 1968')
cars.groupby('Year').describe()


# ## Plots

# In[9]:


fig, ax = plt.subplots()

plt.plot(cars['Month'], cars['Sales'])
plt.title('Car sales between 1960 and 1968')
plt.xlabel('Year')
plt.ylabel('Sales')

labels = [item.get_text() for item in ax.get_xticklabels()]

# this for loop puts a year label at the beginning of each year
for i in range(9):
    split = cars['Month'][i*12].split("-")
    labels[i*12] = split[0]

    
ax.set_xticklabels(labels)
plt.xticks(rotation=70)

plt.show()

# TO DO: it would be better to change strings cars['Month'] to timestamp data                                    


#   
#   
# # Forcasting time series data
# Using the Facebook Prophet to predict future data. For detailed information about `fbprophet` visit https://github.com/facebook/prophet.

# In[10]:


cars.drop(columns=['Year']) # removing column used in summary statistics to prep the df for fbprophet

cars = cars.rename(columns={'Month': 'ds', 'Sales': 'y'}) # to use fbprophet columns must be named 'ds' and 'y'
cars_model = Prophet(interval_width=0.95)
cars_model.fit(cars)

cars_forecast = cars_model.make_future_dataframe(periods=36, freq='MS')
cars_forecast = cars_model.predict(cars_forecast)

plt.figure(figsize=(18, 6))
cars_model.plot(cars_forecast, xlabel = 'Year', ylabel = 'Sales')

plt.title('Car Sales')


plt.show()

# INFO: Prophet automatically detected monthly data and disabled weekly and daily seasonality.


# In[ ]:




