# Imports
import pandas as pd 
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats


# File information

def wrangle_info():
    print(f'This wrangle file has the following functions:')
    print()
    print(f'get_airline_data()')
    print(f'summarize(df)')
    print(f'prep_airline(df)')
    print(f"split_function(df, 'satisfaction')")


#-------------------------------------------------------------------------------------------------------------------


# Initiating csv collection or creation

def get_airline_data(filename="airline.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df
        - write df to csv
    - Output airline passenger satisfaction df
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0) 
        print('Found CSV!')
        return df
    
    else:
        df1 = pd.read_csv('train.csv')
        df2 = pd.read_csv('test.csv')
        df = pd.concat([df1, df2])
        df = df.reset_index(drop=True)
        df.columns = df.columns.str.replace(' ', '_').str.replace('/','_').str.replace('-','_').str.lower()
        df = df.drop(columns='unnamed:_0')
        df = df.rename(columns={'class': 'customer_class'})
        df = df.dropna()
        #want to save to csv
        df.to_csv(filename)
        print('Creating CSV!')
        return df




#-------------------------------------------------------------------------------------------------------------------



# Initializing summary for data frame

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols



def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols


def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing



def nulls_by_row(df, index_id = 'id'):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})

    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True).reset_index()[[index_id, 'num_cols_missing', 'percent_cols_missing']]
    
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)




def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
    num_cols = len(get_numeric_cols(df))
    num_rows, num_cols_subplot = divmod(num_cols, 3)
    if num_cols_subplot > 0:
        num_rows += 1
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
    
    for i, col in enumerate(get_numeric_cols(df)):
        row_idx, col_idx = divmod(i, 3)
        sns.histplot(df[col], ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set_title(f'Histogram of {col}')
    
    plt.tight_layout()
    plt.show()




#-------------------------------------------------------------------------------------------------------------------



# Creating dummy columns for modeling stage

def prep_airline(df):
    '''
    This function will pepare airline dataset for modeling
    '''
    
    dummy_airline = pd.get_dummies(df[['gender', 'customer_type', 'type_of_travel', 
                      'customer_class', 'satisfaction']], drop_first=True)
    df = pd.concat([df, dummy_airline], axis=1)
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df



#-------------------------------------------------------------------------------------------------------------------


#SPLIT FUNCTION

def split_function(df, target_variable):
    '''
    Take in a data frame and returns train, validate, test subset data frames
    Input target_variable as a string
    '''
    train, test = train_test_split(df,
                              test_size=0.20,
                              random_state=123,
                              stratify= df[target_variable]
                                  )
    train, validate = train_test_split(train,
                                  test_size=.25,
                                  random_state=123,
                                   stratify= train[target_variable]
                                      )
    print(f'   og_df: ',df.shape)
    print(f'   Train: ',train.shape)
    print(f'Validate: ', validate.shape)
    print(f'    Test: ',test.shape)
    
    return train, validate, test





