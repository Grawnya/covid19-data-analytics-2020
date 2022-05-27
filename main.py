# import all libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
# read in all relevant csv files
visitorData = 'E:\Google CSV File'
dataVisitors = pd.read_csv(visitorData)
deathData = 'E:\OWID COVID-19 CSV File'
dataDeaths = pd.read_csv(deathData)
mobilityData = 'E:\Apple CSV File'
dataMobility = pd.read_csv(mobilityData)

# to organise the data for any country
def countryData(Country):
 x = Country
 Apple = dataMobility.T
 Apple.columns = Apple.iloc[0,:]
 Apple = Apple.drop(Apple.index[[0,0]])
 Apple = Apple.rename_axis(None, axis = 1)#organise Apple mobility data
 
 Country = Apple['Republic of Korea'] # select country for Apple data
 Country.columns = Country.iloc[0,:]
 Country = Country.drop(Country.index[[0,1]])
 Country[:] -= 100 # set Apple data to the same baseline as Google data
 indexValues = Country.index.values
 Dates = pd.DataFrame(indexValues, columns=['Date']) 
 Country = Country.reset_index(drop=True)
 join = pd.concat([Dates,Country],axis=1) # tidy Apple data for merging
 
 finalCountry = dataVisitors.loc[dataVisitors.Entity == x]#selectGoogle
 Deaths = dataDeaths.loc[dataDeaths.location==x]#select death+case data
 Travel = join.copy() # select the Apple data
 
 merged = pd.merge(finalCountry, Deaths,how='outer') # merge datasets
 merged = merged.drop(['total_cases','location','human_development_index','total_vaccinations',\
     'new_vaccinations_smoothed_per_million','total_vaccinations_per_hundred','new_vaccinations',\
     'new_vaccinations_smoothed','positive_rate','tests_per_case','weekly_hosp_admissions',\
     'weekly_hosp_admissions_per_million','weekly_icu_admissions','weekly_icu_admissions_per_million',\
     'reproduction_rate','hosp_patients','hosp_patients_per_million','icu_patients','icu_patients_per_million',\
     'new_cases_smoothed','new_deaths_smoothed_per_million','new_deaths_smoothed',\
     'new_cases_smoothed_per_million','total_deaths','total_cases_per_million','new_cases_per_million', \
'total_deaths_per_million','new_deaths_per_million', 'total_tests', \
'new_tests','total_tests_per_thousand', \
'new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand', 'tests_units', 'stringency_index', \
'population', 'population_density', 'median_age','aged_65_older', \
'aged_70_older', 'gdp_per_capita', \
'extreme_poverty','cardiovasc_death_rate', 'diabetes_prevalence', \
'female_smokers','male_smokers', 'handwashing_facilities', \
'hospital_beds_per_thousand','life_expectancy' ], 1) # drop columns
 merged = pd.merge(merged, Travel,how='outer')
 merged = merged.loc[0:333]
 merged = merged.replace(np.nan, 0)#tidy merged dataset, filling blanks 
 return merged

# to create graphs of the dataset values, where "diff" refers to days 
def googleGraph(nameOfCountry,yMaximum,Country,day1,Date1,a,day2diff,Date2,b,day3diff,Date3,c,day4diff,Date4,d,day5diff,Date5,e):
 fig = plt.figure(figsize=(20,10))
 ax =fig.add_subplot(1,1,1)
 
 baseDate = Country['Date'][0]
 # plot case and death data
 plt.axvspan(0,day2diff,facecolor='#95c5e8',alpha=0.5)
 plt.axvspan(day2diff,day3diff,facecolor='#8858f6',alpha=0.5)
 plt.axvspan(day3diff,333,facecolor='#f8585a',alpha=0.5)
 ax.bar(Country['Date'],Country['new_cases'],color = '#9d9d9d', label = 'New Cases')
 ax.plot(Country['Date'], Country['new_deaths'], color='#ff0d0d', label 
='New Deaths')
 ax.legend(loc='lower left')
 ax.set_xlim(xmin = Country['Date'][0],xmax=Country['Date'][333])
 ax2 = ax.twinx()
 
 # plot Google mobility data
 ax2.axhline(y=0, color='#d4d2d1', linestyle='-')
 ax.set_ylim(ymin = 0,ymax=Country['new_cases'].max() + 50)
 ax2.plot(Country['Date'], Country['Parks (%)'], 'g', marker = 'o', 
markerfacecolor='#11bd01', markersize=3, label ='Parks')
 ax2.plot(Country['Date'], Country['Retail & Recreation (%)'], 
'#85067d', marker = 'o', markerfacecolor='#85067d', markersize=3, label 
='Retail')
 ax2.plot(Country['Date'], Country['Grocery & Pharmacy Stores (%)'], 
'#e5100d', marker = 'o', markerfacecolor='#e5100d', markersize=3, label 
='Grocery')
 ax2.plot(Country['Date'], Country['Residential (%)'], '#fbbe04', 
marker = 'o', markerfacecolor='#fbbe04', markersize=3, label 
='Residential')
 ax2.plot(Country['Date'], Country['Transit Stations (%)'], '#bbfc16', 
marker = 'o', markerfacecolor='#bbfc16', markersize=3, label ='Transport')
 ax2.plot(Country['Date'], Country['Workplaces (%)'], '#fa0b8b', marker 
= 'o', markerfacecolor='#fa0b8b', markersize=3, label ='Workplace')
 plt.xlabel("Date",fontsize=15) 
 plt.ylabel("% Change",fontsize=15)
 
 Head = yMaximum - 2
 
 ax2.text(Date1, Head-10, 'Special Disaster' ,fontsize=12)# for event 1
 ax2.text(Date1, Head-13, 'Zones Declared' ,fontsize=12)# for event 1
 ax2.text(Date2, Head, b ,fontsize=12)# for event 2
 ax2.text(Date3, Head, 'Seoul' ,fontsize=12)# for event 3
 ax2.text(Date3, Head-3, 'Restrictions' ,fontsize=12)# for event 3
 ax2.text(Date4, Head, d ,fontsize=12)# for event 4
 ax2.text(Date5, Head, e ,fontsize=12)# for event 5
 Title = 'Impact of COVID-19 on ' + nameOfCountry
 plt.figtext(.5,.9,Title, fontsize=20, ha='center')
 ax2.legend(loc='best')
 
 plt.axvline(x=day1, linewidth=2, color='#d4d2d1', linestyle='--')#for full lockdown
 plt.axvline(x=day2diff, linewidth=2, color='#d4d2d1', linestyle='--')#for lockdown extended
 plt.axvline(x=day3diff, linewidth=2, color='#d4d2d1', linestyle='--')#for schools close
 plt.axvline(x=day4diff, linewidth=2, color='#d4d2d1', linestyle='--')#for phase 0
 plt.axvline(x=day5diff, linewidth=2, color='#d4d2d1', linestyle='--')#for good friday and lockdown extended again
 
 ax.set_ylabel('No. of Cases per Day', fontsize=15)
 ticks = ax.set_xticks([baseDate, Date1, Date2, Date3, Date4, Date5])
 labels = ax.set_xticklabels([baseDate, Date1, Date2, Date3, Date4, 
Date5],rotation=30, fontsize='small')
 ax.set_xlabel("Date",fontsize=15)
 return
from datetime import datetime
date_format = "%d/%m/%Y"
# calculate number of days between date and first day
def diffDays(firstDate,date,date_format):
 day = datetime.strptime(date, date_format)
 Sum = day - firstDate
 number = Sum.days
 return number
# country
x = 'Name'
country = countryData(x) # creates dataset for country
firstDate = datetime.strptime('17/02/2020', date_format)
date1 = '01/01/2020' # day 1 info
number1 = diffDays(firstDate,date1,date_format)
title1 = 'Day 1'
date2 = '02/02/2020' # day 2 info
number2 = diffDays(firstDate,date2,date_format)
title2 = 'Day 2'
date3 = '03/03/2020' # day 3 info
number3 = diffDays(firstDate,date3,date_format)
title3 = 'Day 3'
date4 = '04/04/2020' # day 4 info
number4 = diffDays(firstDate,date4,date_format)
title4 = 'Day 4'
date5 = '05/05/2020' # day 5 info
number5 = diffDays(firstDate,date5,date_format)
title5 = 'Day 5'
googleGraph(x,98,country,number1,date1,title1,number2,date2,title2,number3,date3,title3,number4,date4,title4,number5,date5,title5)
dates = [date1,date2,date3,date4,date5]

# generates country statistics
def stats(df,column):
 mean = df[column].mean()
 std = df[column].std()
 Min = df[column].min()
 Max = df[column].max()
 stat =[mean,std,Min,Max]
 return stat
# loops though country categories
Columns = country.columns[2:10]
for column in Columns:
 print(column, ':', stats(country,column))
# correlation of the country
corr = country.corr(method='spearman')
plt.figure(figsize=(10, 8))

ax = sns.heatmap(
 corr, # correlation value r - strength of linear relationship 
 vmin=-1, vmax=1, center=0,
 cmap='coolwarm',
 square=True,
 annot = True, 
 fmt='.2g'
)
ax.set_xticklabels(
 ax.get_xticklabels(),
 rotation=45,
 horizontalalignment='right'
);
plt.figtext(.5,.9,'Correlation of COVID-19 Factors of ' + x, fontsize=15, 
ha='center')

def correlation(mylist,number,title,days):
 
 '''
 Prints off correlation matrix of a lockdown period.
 Creating a certain number of "days" window of the visits compared to 
the subsequent 
 number of "days" of case rises before a lockdown and producing the 
resultant
 subset.
 '''
 
 lockDown = mylist.iloc[number-days:number,2:8] # Gathering Values
 Cases = mylist.iloc[number:number+days,9]
 Deaths = mylist.iloc[number:number+days,10]
 maxTemp = mylist.iloc[lockDown-days:lockDown,13]
 minTemp = mylist.iloc[lockDown-days:lockDown,14]
 Precipitation = mylist.iloc[lockDown-days:lockDown,15]
 lockDown["Cases"]= Cases.values #Adding Values to lockDown DataFrame
 lockDown["Deaths"]= Deaths.values
 lockDown["Max Temp"]= maxTemp.values
 lockDown["Min Temp"]= minTemp.values
 lockDown["Precipitation"]= Precipitation.values
 lockDown.reset_index(drop=True, inplace=True)
 
 corrL = lockDown.corr(method='spearman')
 plt.figure(figsize=(10, 8))
 ax = sns.heatmap(
 corrL, 
 vmin=-1, vmax=1, center=0,
 cmap='coolwarm',
 square=True,
 annot = True, 
 fmt='.2g'
 )
 ax.set_xticklabels(
 ax.get_xticklabels(),
 rotation=45,
 horizontalalignment='right'
 );
 plt.figtext(.5,.9,'Correlation of COVID-19 Factors of ' + x + ' related to ' + title ,fontsize=15, ha='center')
 return lockDown
# event example for correlation matrix and statistics
lag = 14
lockDown1 = correlation(country,number1,title1,lag)
lock1Cols = lockDown1.columns
for column in lock1Cols:
 print(column, ':', stats(lockDown1,column))

def singleLinearRegression(mylist,xAxis,yAxis):
 '''Perform OLS Single Linear Regression'''
 X = mylist.iloc[:, xAxis].values.reshape(-1, 1)#values convert-nparray
 Y = mylist.iloc[:, yAxis].values.reshape(-1, 1) 
 linear_regressor = LinearRegression()
 model = linear_regressor.fit(X, Y) # perform linear regression
 Y_pred = model.predict(X) # make predictions
 
 fig = plt.figure(figsize=(6,4))
 ax =fig.add_subplot(1,1,1)
 plt.scatter(X, Y, c='#12cd74', s=20)
 plt.plot(X, Y_pred, color='red')
 plt.title('Linear Regression Analysis',fontsize=15, ha='center')
 plt.xlabel(mylist.columns[xAxis],fontsize=10) 
 plt.ylabel(mylist.columns[yAxis],fontsize=10) 
 plt.show()
 r_sq = model.score(X, Y)
 
 X2 = sm.add_constant(X)
 est = sm.OLS(Y, X2)
 est2 = est.fit()
 print(est2.summary())
 print('regression coefficients:', est2.params)
 print('R-squared: %.3f' % r_sq) # coefficient of determination r2 
 print('α constant (intercept): %.3f' % model.intercept_) # intercept
 print('β coefficient (slope): %.3f' % model.coef_) # slope of line

# Total OLS Regression Model
Length = [2,3,4,5,6,7,9,10,11,12]
for i in Length:
 singleLinearRegression(country,i,8)
