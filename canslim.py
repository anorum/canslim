
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time
'''
Class - AnnualChange

Takes in dataframe and list of columns and computes the annual change and annual percentage change for the list of columns.

'''

from sklearn.base import BaseEstimator, TransformerMixin

class AnnualChange(BaseEstimator, TransformerMixin):
    def __init__(self, cols, add_percent=True, return_cols = False):
        self.add_percent = add_percent
        self.cols = cols
        self.return_cols = return_cols
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        df = df.sort_values(['symbol','Year', 'Month'],ascending=True)
        perc_cols = []
        change_cols = []
        for col in self.cols:
            #Generate Total Change
            cur = df.set_index(['Year', 'Month'])
            prev = df.set_index(['Year','Month']).groupby(['Month']).shift()
            change = np.subtract(cur[col], prev[col]) \
                .where(prev['symbol'] == cur['symbol'])
            df[str(col) + '_annual'] = change.values
            change_cols.append(str(col)+'_annual')
            if self.add_percent:
            #Generate Percentage Change
                perc_change = np.divide(change,abs(prev[col]))
                df[str(col) + '_annualp'] = perc_change.values
                df[str(col) + '_annualp'] = df[str(col) + '_annualp'].replace(-np.inf,np.nan)
                df[str(col) + '_annualp'] = df[str(col) + '_annualp'].replace(np.inf,np.nan)
                perc_cols.append(str(col) + '_annualp')
        if self.return_cols and self.add_percent:
            print('Column(s) Generated : ' + str(change_cols))
            print('Columns(s) Generated : ' + str(perc_cols))
            return df, change_cols, perc_cols
        elif self.return_cols:
            print('Column(s) Generated : ' + str(change_cols))
            return df, change_cols
        else:
            print('Column(s) Generated : ' + str(change_cols))
            return df
        

'''
Class - QuarterlyChange

Takes in the Dataframe and a list of columns and computes the quarterly change and percent quarterly change for the list of columns.
'''

class QuarterlyChange(BaseEstimator, TransformerMixin):
    def __init__(self, cols, add_percent=True, return_cols = False):
        self.add_percent = add_percent
        self.cols = cols
        self.return_cols = return_cols
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        #Function takes a dataframe and a list of columns to generate the change from current value to the next quarter value
        #Calculates percentage of change as well.
        df = df.sort_values(['symbol','Year','Month'],ascending=True)
        #Return lists of columns created for later analysis
        perc_cols = []
        change_cols = []
        for col in self.cols:
                #Current Col
                cur = df.set_index(['Year','Month'])
                #Shifted Col where the symbol matches
                prev = df.set_index(['Year','Month']).shift()
                change = np.subtract(cur[col], prev[col]) \
                    .where(prev['symbol'] == cur['symbol'])
                
                #Change from current to prev
                df[str(col) + '_qchange'] = change.values
                change_cols.append(str(col)+'_qchange')
                if self.add_percent:
                    #Need to handle zero change values
                    perc = np.divide(change,abs(prev[col]))
                    df[str(col) + '_qperc'] = perc.values
                    df[str(col) + '_qperc'] = df[str(col) + '_qperc'].replace(-np.inf,np.nan)
                    df[str(col) + '_qperc'] = df[str(col) + '_qperc'].replace(np.inf,np.nan)
                    perc_cols.append(str(col) + '_qperc')
        if self.return_cols and self.add_percent:
            print('Column(s) Generated : ' + str(change_cols))
            print('Columns(s) Generated : ' + str(perc_cols))
            return df, change_cols, perc_cols
        elif self.return_cols:
            print('Column(s) Generated : ' + str(change_cols))
            return df, change_cols
        else:
            print('Column(s) Generated : ' + str(change_cols))
            return df


'''
Function - Get Price Change and Shift to Make Target Column

Runs same function as Quarterly and Annual Change , but instead shifts the price change column up as the outcome. This will allow us to use it as a target column
'''

class PriceTarget(BaseEstimator, TransformerMixin):
    def __init__(self, add_annual=False):
        self.add_annual=add_annual
        self.cols = ['Price']
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        qc = QuarterlyChange(cols=['Price'])
        df = qc.fit_transform(df)
        df['Price target'] = df['Price_qchange'].shift(-1)
        df['Pricep target'] = df['Price_qperc'].shift(-1)
        df.drop(['Price_qchange', 'Price_qperc'],axis=1,inplace=True)
        if self.add_annual:
            ac = AnnualChange(df,cols=['Price'])
            df = ac.fit_transform(df)
            df['Price_a target'] = df['Price_annual'].shift(-1)
            df['Price_a perc'] = df['Price_annualp'].shift(-1)
            df.drop(['Price_annual','Price_annualp'],axis=1,inplace=True)
            print ("Targets Created : ['Price_a target', 'Price_a perc']")
        print("Targets Created : ['Price target','Pricep target']")
        return df


'''
Class - AccelIncrease

Calculates the streak of which the fundamentals percentage change is increasing.

For example a fundamental that increased by 20% the 1st quarter and by 22% the 2nd quarter and 10% the 3rd quarter would have a AccelIncrease column as 1, 2, 0
'''

class AccelIncrease(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        change_col = []
        for col in self.cols:
            df[str(col) + '_qacceleration'] = (df[col] - df[col].shift())
            df['increase'] = (df[col] - df[col].shift()) > (df[col] - df[col].shift()).shift()
            df[str(col) + '_accstreak'] = df['increase'].groupby(((df['increase']!=df['increase'].shift())).cumsum()).cumsum() \
            .where(df['symbol'] == df['symbol'].shift())
            df.drop('increase',axis=1,inplace=True)
            change_col.append(str(col) + '_qacceleration')
            change_col.append(str(col) + '_accstreak')
        print('Columns Created : ' + str(change_col))
        return df


'''
Class - StreakIncrease

Calculates the number of streaks in which a fundamental has been increasing
'''

class StreakIncrease(BaseEstimator, TransformerMixin):
    def __init__(self,cols, x):
        self.cols = cols
        self.x = x
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        change_col = []
        for col in self.cols:
            df['gt'] = df[col] > self.x
            df[str(col) + '_streakgt_' + str(self.x)] = df['gt'].groupby(((df['gt']!=df['gt'].shift())).cumsum()).cumsum() \
            .where(df['symbol'] == df['symbol'].shift())
            df.drop('gt',axis=1,inplace=True)
            change_col.append(str(col) + '_streakgt_' + str(self.x))
        print('Columns Created : ' + str(change_col))
        return df



'''
Function - outliers_z_score

Takes in a vector and then filters out any datapoints that are not within input thres standard deviations from the mean.
'''

def outliers_z_score(ys, thres=2):
    threshold = thres

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.abs(z_scores) < threshold


'''
Function - Explore

Function for quick distribution exploring given a column. Target column will be Price Percentage. Also takes arguments to filter out outliers using outliers_z_score function for x and y.
'''
def explore(df, col, youtliers=True, ythres=2, xoutliers=True, xthres=2):
    if xoutliers and youtliers:
        null_c = df[col].isnull().sum()
        c = df[col].shape[0]
        null_p = round((null_c/c)*100,2)
        print('Null Count: ', null_c, '(', null_p,'%)' )
        print(df[col].describe())
        sns.jointplot(data=df, y='Pricep target', x=col, kind='reg', color='g')
        #plt.axvline(df[col].mean(),c='y')
        #plt.axvline(df[col].median(),c='r')
        plt.show()
    elif (youtliers == False) and xoutliers:
        d = df[outliers_z_score(df['Pricep target'],thres=ythres)]
        null_c = d[col].isnull().sum()
        c = d[col].shape[0]
        null_p = round((null_c/c)*100,2)
        print('Null Count: ', null_c, '(', null_p,'%)' )
        print(d[col].describe())
        sns.jointplot(data=d, y='Pricep target', x=col, kind='reg', color='g')
        #plt.axvline(d[col].mean(),c='y')
        #plt.axvline(d[col].median(),c='r')
        plt.show()
    elif youtliers and (xoutliers == False):
        d = df[outliers_z_score(df[col],thres=xthres)]
        null_c = d[col].isnull().sum()
        c = d[col].shape[0]
        null_p = round((null_c/c)*100,2)
        print('Null Count: ', null_c, '(', null_p,'%)' )
        print(d[col].describe())
        sns.jointplot(data=d, y='Pricep target', x=col, kind='reg', color='g')
        #plt.axvline(d[col].mean(),c='y')
        #plt.axvline(d[col].median(),c='r')
        plt.show()
    else:
        d = df[(outliers_z_score(df[col],thres=xthres)) & (outliers_z_score(df['Pricep target'],thres=ythres))]
        null_c = d[col].isnull().sum()
        c = d[col].shape[0]
        null_p = round((null_c/c)*100,2)
        print('Null Count: ', null_c, '(', null_p,'%)' )
        print(d[col].describe())
        sns.jointplot(data=d, y='Pricep target', x=col, kind='reg', color='g')
        #plt.axvline(d[col].mean(),c='y')
        #plt.axvline(d[col].median(),c='r')
        plt.show()
