# Imports
import pandas as pd 
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats




# Explore!!!!!

def get_distance_satisfaction_ttest(train):
    '''
    insert train df for flight distance and satisfaction ttest
    '''
    # setting alpha (confidence level)
    alpha = 0.05
    
    # isolating satisfied customers for ttest
    satisfied = train[train.satisfaction == 'satisfied']
    
    # calculating mean flight distance for ttest
    overall_mean = train.flight_distance.mean()
    
    # initiating ttest
    t, p = stats.ttest_1samp(satisfied.flight_distance, overall_mean)
    
    # print values
    print(f't = {t:.4}')
    print(f'p = {p/2:.4}')
    
    # printing test outcome results
    if p/2 > alpha:
        print('We fail to reject the null hypothesis')
    elif t < 0:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the hypothesis')


# Chi^2 test for categorical vs categorical 


def get_chi2_results(train, var1, var2):
    '''
    insert train df and 2 cat vars as strings for chi^2 test
    '''

    #set alpha
    alpha = 0.05

    # create observed for test
    observed = pd.crosstab(train[var1], train[var2])

    # setting up chi^2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # print the chi2 value, formatted to a float with 4 digits
    print(f'chi^2 = {chi2:.4f}')
    
    # print the p-value, formatted to a float with 4 digits
    print(f'p.    = {p:.4f}')
    
    # print the result of the test
    if p > alpha:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the null hypothesis')
    return observed



























