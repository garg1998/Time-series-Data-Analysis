#!/usr/bin/env python
# coding: utf-8

# ### About the Data
# There are two cities, San Juan and Iquitos, with weather/temperature data for each city spanning 5 and
# 3 years respectively. The data for each city have been concatenated along with a city column indicating
# the source: sj for San Juan and iq for Iquitos. Throughout the dataset, missing values have been filled as
# NaNs.
# 
# * city – Temperature observations locations
# * year – Year of the temperature observation
# * weekofyear – Week of year of the temperature observation
# * week_start_date – Data of a week start of the temperature observation
# * station_max_temp_c – Maximum temperature
# * station_min_temp_c – Minimum temperature
# * station_avg_temp_c – Average temperature
# * station_precip_mm – Total precipitation
# * station_diur_temp_rng_c – Diurnal temperature range
#      

# In[475]:


# importing all necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for visualization
import matplotlib.pyplot as plt #for visualization

import warnings
warnings.filterwarnings("ignore")


# ### Importing Data

# In[405]:


# Importing data
df=pd.read_csv('flexgen_data_analysis_test.csv')
df.head()


# In[406]:


#checking data types of each column
df.info()


# In[400]:


df.info()


# In[401]:


df.shape


# ### Seperating Data set according to the city

# In[407]:


df_sj=df[df['city']=='sj']
df_iq=df[df['city']=='iq']


# ### checking for missing values

# In[408]:


df_sj.isna().sum()


# In[409]:


df_iq.isna().sum()


# ### Interpolation 

# In[410]:


df_sj=df_sj.interpolate(limit_direction='both')


# In[411]:


df_iq=df_iq.interpolate(limit_direction='both')


# In[412]:


df_sj['week_start_date']=pd.to_datetime(df_sj['week_start_date'])
df_iq['week_start_date']=pd.to_datetime(df_iq['week_start_date'])


# # Q.1 EDA

# ### San Juan 

# In[413]:


df_sj.describe()


# In[414]:


#temperature distributions in histogram
plt.figure(figsize=(8,6))
df_sj.station_max_temp_c.hist(alpha=0.6,label='max_Temp')
df_sj.station_min_temp_c.hist(alpha=0.6, label='min_temp')
df_sj['station_avg_temp_c'].hist(alpha=0.7,label='avg_temp')

plt.title("Histogram of Avg. Temp, Min. Temp, Max Temp.")
plt.xlabel("Temp.")

plt.legend()
plt.show()


# * According to Min temp. distribution, most data lies between 21 and 25 degree celcius
# * avg temperature has a bimodal behaviour, probably indicating peak tempurature across summer and winter season
# * According to Max temp. distribution, most data lies between 30 and 33 degree celcius

# In[415]:


# Histogram of Diuranl Temp. range
plt.figure(figsize=(8,6))
df_sj.station_diur_temp_rng_c.hist(alpha=0.6,label='Diuranl Temp. range')
plt.title("Histogram of Diuranl Temp. range")
plt.xlabel("Temp.")

plt.legend()
plt.show()


# * The temp. difference b/w min and max mostly fluctuate b/w 6.5 degree celcius

# In[416]:


df_sj['month']=df_sj['week_start_date'].dt.month


# In[417]:


# summer in San Juan ~ June- Aug ~ 6 to 8 Months
# Winter in San Juan ~ December to Feb ~ 12 to 2 months


# In[418]:


summer=df_sj[df_sj['month'].isin([6, 7, 8])]
winter=df_sj[df_sj['month'].isin([12, 1, 2])]


# In[419]:


plt.figure(figsize=(8,6))
summer.station_avg_temp_c.hist(bins=20,alpha=0.6,label='summer')
winter.station_avg_temp_c.hist(bins=20,alpha=0.6, label='winter')
plt.legend()
plt.show()


# * For Winter months, most temp points lines 25 to 27 degree celcius with peak at 26 
# * For Summer months, most temp points lines 28 to 30 degree celcius with peak at 29

# In[420]:


# Analyzing Temp. flucations across months
grouped_mths=df_sj.groupby(df_sj.month)[['station_max_temp_c', 'station_min_temp_c','station_avg_temp_c']].agg([min, max])
grouped_mths['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
grouped_mths=grouped_mths.set_index('months')
grouped_mths


# 
# * least max temp: 27.2 in Jan  and highest max temp: 35 in June, July and sept. 
# * least min temp: 20 in Dec, Feb, Mar and highest min temp: 22.8 in July, Aug, oct
# 
# * least avg temp: 24.1 Jan and highest max temp: 30.27 July

# In[421]:


# total precipitation distribution
plt.figure(figsize=(8,6))
df_sj.station_precip_mm.hist(alpha=0.6,label='total_precipitation')
plt.legend()
plt.show()


# The distribution is highly skewed towards the right. Most of the precipitation occurs <50mm

# In[422]:


rain_df=df_sj.groupby(['month'])[['station_precip_mm']].agg([min, max, np.mean]).reset_index()
rain_df['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
rain_df=rain_df.set_index('months')
rain_df


# In[423]:


# bar plot of avg. rain across months
rain_df['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x = 'months',y = rain_df['station_precip_mm']['mean'],data = rain_df).set(title='Avg Rainfall across months')
 # Show the plot
plt.show()


# * Highest avg rainfall is recieved in June. 
# * Max precipitation is recieved in Jan with 207.7 mm

# ### Iquitos

# In[424]:


df_iq.describe()


# In[425]:


#temperature distributions in histogram
plt.figure(figsize=(8,6))
df_iq.station_max_temp_c.hist(alpha=0.6,label='max_Temp')
df_iq.station_min_temp_c.hist(alpha=0.6, label='min_temp')
df_iq['station_avg_temp_c'].hist(alpha=0.7,label='avg_temp')
plt.title("Histogram of Avg. Temp, Min. Temp, Max Temp.")
plt.xlabel("Temp.")
plt.legend()
plt.show()


# In[426]:


df_iq.station_diur_temp_rng_c.hist(alpha=0.6, label='Diuranl Temp. range')
plt.title("Histogram of Diuranl Temp. range")
plt.xlabel("Temp.")

plt.legend()
plt.show()


# * Min temp graph is left skewed with most of the min temp  > 20 degree celcius
# * Avg temp graph is left skewed with most of the temp b/w 26 and 29 degree celcius
# * Max temp is somewhat normall distributed with mast of the temp around 35 degree celcius
# * Temp. difference in Iquitos is much higher than San Juan fluctuating around 10 to 12 degree celcius

# In[427]:


df_iq['month']=df_iq['week_start_date'].dt.month


# In[428]:


grouped_mths=df_iq.groupby(df_iq.month)[['station_max_temp_c', 'station_min_temp_c','station_avg_temp_c']].agg([min, max])
grouped_mths['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
grouped_mths=grouped_mths.set_index('months')
grouped_mths


# 
# * least max temp: 30.8 in May and highest max temp: 38.4 in Oct. 
# * least min temp: 14.2 in Jul and highest min temp: 23.2 in Jan
# * Max and min temp. of Avg Temp across all the months
# * least avg temp: 24.8 Jun and highest max temp: 29.1 Oct
# 
# **Summer in Iquitos (November – May) is rainy, warm and humid. In May, the Amazon river that surrounds Iquitos reaches its highest point. Winter in Iquitos offers a different climate, sunny days and nice weather, with an average of 90 F (32). The dry season (July – November)**

# In[429]:



avg_df=df_iq.groupby(df_iq.month)[['station_avg_temp_c']].agg([min, max])
avg_df['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
avg_df=avg_df.set_index('months')
avg_df


# In[430]:


# total precipitation distribution
plt.figure(figsize=(8,6))
df_iq.station_precip_mm.hist(alpha=0.6,label='total_precipitation')
plt.legend()
plt.show()


# In[431]:


rain_df_iq=df_iq.groupby(df_iq.month)[['station_precip_mm']].agg([min, max, np.mean])
rain_df_iq['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
rain_df_iq=rain_df_iq.set_index('months')
rain_df_iq


# In[434]:


rain_df_iq['months'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x = 'months',y = rain_df_iq['station_precip_mm']['mean'],data = rain_df_iq).set(title='Avg Rainfall across months')
 # Show the plot
plt.show()


# * The distribution is highly skewed towards the right. Most of the precipitation occurs <50mm
# * Highest avg rainfall is recieved in Mar. 
# * Max precipitation is recieved in Apr with 212 mm

# ### Q2. Build graphical representations of the analyzing dataset: weekly (min, max, avg)

# In[435]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,8.27))

# Plot the first sns graph on the left subplot
sns.lineplot(data=df_sj,x='week_start_date',y='station_avg_temp_c',ax=ax1)
ax1.set_title('Avg Temp across weeks - San Juan')

# Plot the second sns graph on the right subplot
sns.lineplot(data=df_iq,x='week_start_date',y='station_avg_temp_c',ax=ax2)
ax2.set_title('Avg Temp across weeks - Iquitos')

# Display the plots
plt.show()


# * San Juan: Horizontal trend is observed and a strong seasonality occuring every year
# * Iquitos: Downward trend is observed and a seasonality occuring every year

# In[436]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,8.27))

# Plot the first sns graph on the left subplot
sns.lineplot(data=df_sj,x='week_start_date',y='station_max_temp_c',ax=ax1)
ax1.set_title('Max Temp. across weeks - San Juan')

# Plot the second sns graph on the right subplot
sns.lineplot(data=df_iq,x='week_start_date',y='station_max_temp_c',ax=ax2)
ax2.set_title('Max Temp. across weeks - Iquitos')

# Display the plots
plt.show()


# * San Juan : Horizontal trend is observed and seasonal pattern can be seen repeating every year
# * Iquitos : Downward trend can be observed 

# In[547]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,8.27))

# Plot the first sns graph on the left subplot
sns.lineplot(data=df_sj,x='week_start_date',y='station_min_temp_c',ax=ax1)
ax1.set_title('Min Temp. across weeks - San Juan')

# Plot the second sns graph on the right subplot
sns.lineplot(data=df_iq,x='week_start_date',y='station_min_temp_c',ax=ax2)
ax2.set_title('Min Temp. across weeks - Iquitas')

# Display the plots
plt.show()


# * San Juan : Horizontal trend is observed and seasonal pattern can be seen repeating every year
# * Iquitos : Upward trend can be observed 

# ## Q.2 Identify annual monthly seasonality and trends of the presented temperature observations based on average temperature

# ### San Juan

# In[438]:


df_sj.head()


# In[439]:


avg_df_sj=df_sj.groupby(['year','month'])[['station_avg_temp_c']].mean().reset_index()
avg_df_sj['year_mn'] = pd.to_datetime(avg_df_sj['year'].astype(str) + ' ' + avg_df_sj['month'].astype(str) + ' 1',
                                format='%Y %m %d')


# In[440]:


avg_df_sj[avg_df_sj['year']==2009]


# In[441]:


avg_df_sj.head()


# In[442]:


sns.set(rc={'figure.figsize':(11,6.27)})
sns.lineplot(data=avg_df_sj,x='year_mn',y='station_avg_temp_c')


# * Seasonality is observed occuring every year with a horizonatal trend

# In[443]:


# Observing rain pattern in 2011
rain_df=df_sj.groupby(['year','month'])[['station_precip_mm']].agg([min, max, np.mean]).reset_index()

rain_df[rain_df['year']==2011]


# ### Iquitos

# In[444]:


avg_df_iq=df_iq.groupby(['month','year'])[['station_avg_temp_c']].mean().reset_index()
avg_df_iq['year_mn'] = pd.to_datetime(avg_df_iq['year'].astype(str) + ' ' + avg_df_iq['month'].astype(str) + ' 1',
                                format='%Y %m %d')


# In[445]:


avg_df_iq[avg_df_iq['year']==2012]


# In[446]:


sns.set(rc={'figure.figsize':(8,6)})
plt.yticks([25,26,27,28,29])
sns.lineplot(data=avg_df_iq,x='year_mn',y='station_avg_temp_c')
plt.title(" Avg Temp. across months - Iquitos")
plt.xlabel("Temp.")

plt.show()


# * Downward trend is observed and a strong seasonality occuring every year

# In[367]:


rain_df=df_iq.groupby(['year','month'])[['station_precip_mm']].agg([min, max, np.mean]).reset_index()

rain_df[rain_df['year']==2012]


# ## Q. 4 Correlation Analysis

# ### San Juan

# In[476]:


sns.set(rc={'figure.figsize':(8,6)})
sns.scatterplot(x="station_avg_temp_c", y="station_precip_mm", data=df_sj);
plt.title(" weekly avg temp. vs total rainfall - San Juan")
plt.xlabel("Temp.")

plt.show()


# In[548]:


df_sj['station_avg_temp_c'].corr(df_sj['station_precip_mm'])


# In[478]:


sj_data=df_sj[['year','station_precip_mm','station_avg_temp_c']]


# In[479]:


year_sj=df_sj['year'].unique()
arr=[]
for y in year_sj:
    arr.append(len(sj_data[sj_data['year']==y]))
arr


# In[480]:


# Function to calculate correlation for all window sizes starting from 1 to w for a particular year 
def window_fx(year,df,w):
    data=df[df['year']==year]
    window=w
    corr_list=[]
    for i in range(2,window+1):
        data['window_temperature_avg']=((data['station_avg_temp_c'].rolling(min_periods=1,window=i).sum()-data['station_avg_temp_c']))/(i-1)
        corr_list.append([year,i,data['window_temperature_avg'].corr(data['station_precip_mm'])])
        
    return corr_list


# In[481]:


year_sj=df_sj['year'].unique()

data=df_sj[['year','station_avg_temp_c','station_precip_mm']]
w=33

lst=[]
for y in year_sj:
    df=window_fx(y,data,w)
    lst.append(df)

sj_data=pd.DataFrame(lst[0],columns=['year','window','corr'] )
for i in range(1,len(lst)):
    d=pd.DataFrame(lst[i],columns=['year','window','corr'] )
    sj_data=sj_data.append(d,ignore_index=True)


# In[482]:


sns.lineplot(data=sj_data, x='window',y='corr',hue='year')
plt.legend()
plt.title(" correlation plot across different weekly temp. windows - San Juan")


# In[483]:


rain=df_sj.groupby(['year','month'])[['station_precip_mm','station_avg_temp_c']].agg([min, max, np.mean]).reset_index()

rain['station_avg_temp_c']['mean'].corr(rain['station_precip_mm']['mean'])


# In[484]:


sns.set(rc={'figure.figsize':(8,6)})
sns.scatterplot(x=rain['station_avg_temp_c']['mean'], y=rain['station_precip_mm']['mean'], data=rain);
plt.title(" monthly avg temp. vs monthly avg. rainfall - San Juan")
plt.xlabel("Temp.")


# ## Iquitos

# In[485]:


sns.set(rc={'figure.figsize':(8,6)})
sns.scatterplot(x="station_avg_temp_c", y="station_precip_mm", data=df_iq);
plt.title(" weekly avg temp. vs total rainfall - Iquitos")
plt.xlabel("Temp.")

plt.show()


# In[486]:


df_iq['station_avg_temp_c'].corr(df_iq['station_precip_mm'])


# In[492]:


iq_data=df_iq[['year','station_precip_mm','station_avg_temp_c']]


# In[493]:


year_iq=df_iq['year'].unique()
arr=[]
for y in year_iq:
    arr.append(len(iq_data[iq_data['year']==y]))
arr


# In[494]:



w=26

lst=[]
for y in year_iq:
    df=window_fx(y,iq_data,w)
    lst.append(df)

iq_data=pd.DataFrame(lst[0],columns=['year','window','corr'] )
for i in range(1,len(lst)):
    d=pd.DataFrame(lst[i],columns=['year','window','corr'] )
    iq_data=iq_data.append(d,ignore_index=True)


# In[495]:


sns.lineplot(data=iq_data, x='window',y='corr',hue='year')
plt.legend()
plt.title(" correlation plot across different weekly temp. windows - Iquitos")


# In[540]:


iq_data=df_iq[['year','station_precip_mm','station_avg_temp_c']]
iq_2013=df_iq[df_iq['year']==2011]


iq_2013['window_temperature_avg']=((iq_2013['station_avg_temp_c'].rolling(min_periods=1,window=2).sum())-iq_2013['station_avg_temp_c'])/(1)


# In[541]:


iq_2013['window_temperature_avg'].corr(iq_2013['station_precip_mm'])


# In[496]:


rain_iq=df_iq.groupby(['year','month'])[['station_precip_mm','station_avg_temp_c']].agg([min, max, np.mean]).reset_index()

rain_iq['station_avg_temp_c']['mean'].corr(rain_iq['station_precip_mm']['mean'])


# In[497]:


sns.set(rc={'figure.figsize':(8,6)})
sns.scatterplot(x=rain_iq['station_avg_temp_c']['mean'], y=rain_iq['station_precip_mm']['mean'], data=rain);
plt.title(" monthly avg temp. vs monthly avg. rainfall - Iquitos")
plt.xlabel("Temp.")


# In[ ]:




