import sys

import pandas as pd

def get_samples_DeCock(df, cols, force=True):
    """
    Return training and validation data samples using pandas's
    random sampling and Dean De Cock suggested method.

    De Cock states the following:

    "The two data sets can be easily created by randomizing the original data 
    and selecting the relevant proportion for each component with the only real
    requirement being that the number of observations in the training set be six 
    to ten times the number of variables."
    
    when force=True the training dataset lower limit is six times the number
    of variables (included), when false, the lower limit is the number
    of variables.

    NOTE: This function could be improved by relaxing the min_validation_size within an 
    interval.
    """
    max_factor = 10
    min_factor = 6

    # calculate number of rows
    rows = len(df)

    # this algorithm begins trying with max factor and substracts one until
    # it finds the mentioned proportion of training/validation data.
    # Always makes sure validation data size is not less than 20% of the data.    
    factor = max_factor
    min_validation_size = round(rows/5)

    if min_validation_size == 0:
        # dont waste your time.
        raise ValueError("Dataset too small.")

    while (force and factor >= min_factor) or (factor > 0):
        training_size = cols*factor
        validation_size = rows - training_size
        
        if (validation_size >= min_validation_size):
            df_validation = df.sample(n=validation_size)
            df_training = df.sample(n=training_size)
            return df_training, df_validation
        else:
            factor -= 1
            continue
    
    raise ValueError("Dataset too small.")

def drop_column(df, column):
    """Drop from dataframe 'column'

    @param df dataframe to drop column
    @param column column to be dropped
    @return dataframe with dropped column
    """
    try:
        df = df.drop(column, axis=1)
    except ValueError as err:
        print(err)

    return df

def fix_missing_with_mode(df):
    """Fixes missing value from all columns using the mode.
    
    @param df dataframe
    @return dataframe    
    """
    return df.fillna(df.mode().iloc[0])

def dummies(df, list=None):
    """Set dummie variables to the dataframe
    for the columns stated in 'list parameter
    
    @param df dataframe
    @list list of columns to set dummies
    @return dataframe with dummies variables
    """
    return pd.get_dummies(df, columns=list)

def read_file(filename):
    """Reads file and process it using panda dataframes.
    
    @param name of the file
    @return dataframe
    """
    try:
        df = pd.read_csv(filename)
        return df
    except IOError:
        print('File "%s" could not be read' % filename)
        sys.exit()

def do_join(df1, df2, index, how):
    """Joins two dataframes by 'index' key they have in
    common.

    @param df1 dataframe with column 'index' present 
    @param df2 another dataframe with column 'index' present
    @index column that will be used to join

    @return dataframe of joined df1 and df2
    """
    return df1.join(df2.set_index(index), how=how, on=index)

def join_by_index(file1, file2, index, how='left'):
    """Joins two csv files by 'index' key they have in
    common.

    @param file1, file2 csv files
    @index column that will be used to join

    @return dataframe of joined df1 and df2
    """
    df1 = read_file(file1)
    df2 = read_file(file2)

    return do_join(df1, df2, index, how)

def join_to_csv(df, output):
    """Writes a csv file with joined dataframe

    @param df dataframe that has been previously joined
    @param output name of the csv file to be written
    """
    return df.to_csv(output)
