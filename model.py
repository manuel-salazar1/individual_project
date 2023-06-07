# Imports
import pandas as pd 
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler




# Create X_train, X_validate, X_test, y_train, y_validate, y_test

drop_cols = ['id', 'gate_location', 'leg_room_service']

def Xy_train_val_test(train, validate, test, target_variable):
    """
    input train, validate, test, after using split function()
    input target_variable as string
    drop_cols formatted as: ['col1', 'col2', 'etc'] for multiple columns
        This function will drop all 'object' columns. Identify additional 
        columns you want to drop and insert 1 column as a string or multiple
        columns in a list of strings.
    X_train, X_validate, X_test, y_train, y_validate, y_test
    """
    
    baseline_accuracy = train[target_variable].value_counts().max() / train[target_variable].value_counts().sum()
    print(f'Baseline Accuracy: {baseline_accuracy:.2%}')
    
    X_train = train.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_validate = validate.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_test = test.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    
    y_train = train[target_variable]
    y_validate = validate[target_variable]
    y_test = test[target_variable]
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test




# Scale data for model


def scaled_df(X_train, X_validate, X_test):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    X_train (pandas DataFrame): The training data.
    X_validate (pandas DataFrame): The validation data.
    X_test (pandas DataFrame): The test data.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_validate_scaled (pandas DataFrame): The scaled validation data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        
    """

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler and use on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Convert the array to a DataFrame
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns, index=X_validate.index)
    
    # Convert the array to a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled





# MODELS!!!!!!!!!!!!!!!!!!!!!!!!!

# KNN 

def k_nearest2(X_train_scaled, y_train, X_validate_scaled, y_validate):
    """
    Trains and evaluates KNN models for different values of k and plots the results.
    Parameters:
    -----------
    X_train: array-like, shape (n_samples, n_features)
        Training input samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training input samples.
    X_validate: array-like, shape (n_samples, n_features)
        Validation input samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation input samples.
    Returns:
    --------
    results: pandas DataFrame
        Contains the train and validation accuracy for each value of k.
    """
    metrics = []
    train_score = []
    validate_score = []
    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_score.append(knn.score(X_train_scaled, y_train))
        validate_score.append(knn.score(X_validate_scaled, y_validate))
        diff_score = train_score[-1] - validate_score[-1]
        metrics.append({'k': k, 'train_score': train_score[-1], 'validate_score': validate_score[-1], 'diff_score': diff_score})
    baseline_accuracy = y_train.value_counts().max() / y_train.value_counts().sum()
    results = pd.DataFrame.from_records(metrics)
    # modify the last few lines of the function
    # drop the diff_score column before plotting
    results_for_plotting = results.drop(columns=['diff_score'])
    with sns.axes_style('whitegrid'):
        ax = results_for_plotting.set_index('k').plot(figsize=(16,9))
    plt.ylabel('Accuracy')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xticks(np.arange(0,21,1))
    min_diff_idx = np.abs(results['diff_score']).argmin()
    min_diff_k = results.loc[min_diff_idx, 'k']
    min_diff_score = results.loc[min_diff_idx, 'diff_score']
    ax.axvline(min_diff_k, linestyle='--', linewidth=2, color='red', label=f'min diff at k={min_diff_k} (diff={min_diff_score:.3f})')
    plt.fill_between(results['k'], train_score, validate_score, alpha=0.2, color='gray', where=(results['k'] > 0))
    plt.title('K Nearest Neighbor', fontsize=18)
    plt.legend()
    plt.show()
    return results



# Decision Tree

def decision_tree(X_train_scaled, X_validate_scaled, y_train, y_validate):
    """
    This function trains a decision tree classifier on the provided training data, and evaluates its performance on the
    validation data for different values of the 'max_depth' hyperparameter. It then generates a plot of the training and
    validation accuracy scores as a function of 'max_depth', and returns a DataFrame containing these scores.
    Parameters:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.
    Returns:
    - scores_df (pandas.DataFrame): A DataFrame containing the training and validation accuracy scores, as well as the
      difference between them, for different values of the 'max_depth' hyperparameter.
    """
    # get data
    scores_all = []
    for x in range(1,20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_train_scaled, y_train)
        train_acc = tree.score(X_train_scaled,y_train)
        val_acc = tree.score(X_validate_scaled, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])
    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc','val_acc','score_diff'])
    # Plot the results
    sns.set_style('whitegrid')
    plt.plot(scores_df['max_depth'], scores_df['train_acc'], label='Train score')
    plt.plot(scores_df['max_depth'], scores_df['val_acc'], label='Validation score')
    plt.fill_between(scores_df['max_depth'], scores_df['train_acc'], scores_df['val_acc'], alpha=0.2, color='gray')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()
    return scores_df



# Random Forest


def random_forest_scores(X_train_scaled, y_train, X_validate_scaled, y_validate):
    """
    Trains and evaluates a random forest classifier with different combinations of hyperparameters. The function takes in
    training and validation datasets, and returns a dataframe summarizing the model performance on each combination of
    hyperparameters.
    Parameters:
    -----------
    X_train : pandas DataFrame
        Features of the training dataset.
    y_train : pandas Series
        Target variable of the training dataset.
    X_validate : pandas DataFrame
        Features of the validation dataset.
    y_validate : pandas Series
        Target variable of the validation dataset.
    Returns:
    --------
    df : pandas DataFrame
        A dataframe summarizing the model performance on each combination of hyperparameters.
    """
    #define variables
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth,random_state=123)
        rf.fit(X_train_scaled, y_train)
        train_score = rf.score(X_train_scaled, y_train)
        validate_score = rf.score(X_validate_scaled, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]
    #Put results into a dataframe
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'diff_score': diff_scores})
    # Set plot style
    sns.set_style('whitegrid')
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(max_depth_values, train_scores, label='train', marker='o', color='blue')
    plt.plot(max_depth_values, validate_scores, label='validation', marker='o', color='orange')
    plt.fill_between(max_depth_values, train_scores, validate_scores, alpha=0.2, color='gray')
    plt.xticks([2,4,6,8,10],['Leaf 9 and Depth 2','Leaf 7 and Depth 4','Leaf 5 and Depth 6','Leaf 3 and Depth 8','Leaf 1and Depth 10'], rotation = 45)
    plt.xlabel('min_samples_leaf and max_depth', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Random Forest Classifier Performance', fontsize=18)
    plt.legend(fontsize=12)
    plt.show()
    return df


# Logistic Regression


def plot_logistic_regression(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''
    Trains multiple logistic regression models with different regularization strengths (C) on the given training
    data, and plots the resulting train and validation scores against C values. The optimal value of C is marked
    by a vertical red dashed line, and the associated difference between the train and validation scores is shown
    in the plot legend.
    Parameters:
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    X_validate : array-like of shape (n_samples, n_features)
        The validation input samples.
    y_train : array-like of shape (n_samples,)
        The target values for training.
    y_validate : array-like of shape (n_samples,)
        The target values for validation.
    Returns:
    df1 : pandas DataFrame
        A table containing the C, train_score, validate_score, and diff_score values for each model.
    '''
    train_scores = []
    val_scores = []
    c_values = [.01, .1, 1, 10 , 100, 1000]
    for c in c_values:
        logit = LogisticRegression(C=c, random_state=123)
        logit.fit(X_train_scaled, y_train)
        train_score = logit.score(X_train_scaled, y_train)
        val_score = logit.score(X_validate_scaled, y_validate)
        train_scores.append(train_score)
        val_scores.append(val_score)
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - val_score for train_score, val_score in zip(train_scores, val_scores)]
    # Put results into a list of tuples
    results = list(zip(c_values, train_scores, val_scores, diff_scores))
    # Convert the list of tuples to a Pandas DataFrame
    df1 = pd.DataFrame(results, columns=['C', 'train_score', 'validate_score', 'diff_score'])
    # Plot the results
    plt.plot(c_values, train_scores, label='Train score')
    plt.plot(c_values, val_scores, label='Validation score')
    min_diff_idx = np.abs(diff_scores).argmin()
    min_diff_c = results[min_diff_idx][0]
    min_diff_score = results[min_diff_idx][3]
    plt.axvline(min_diff_c, linestyle='--', linewidth=2, color='red', label=f'min diff at C={min_diff_c} (diff={min_diff_score:.3f})')
    plt.fill_between(c_values, train_scores, val_scores, alpha=0.2, color='gray')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Accuracy vs C')
    plt.legend()
    plt.show()
    return df1

















