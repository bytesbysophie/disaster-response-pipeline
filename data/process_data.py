import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Paramters:
        message_filepath (str): The filepath of the message csv file.
        categories_filepath (str): The filepath of the categories csv file.
    
    Returns:
        DataFrame: A merged dataframe, including the messages and categories data.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    '''
    Parameters:
        df (DataFrame): A pandas dataframe including messages and a column with all categories and their values.

    Returns:
        DataFrame: A cleaned dataframe with seperate columns for every category, holding the corresponding value.
    
    This function:
        1. creates a dataframe with seperate columns for every category
        2. replaces the single categorie column in the original df with those new columns
        3. removes duplicates.
    '''

    # Create a dataframe with a single column for every category
    categories = df['categories'].str.split(';', expand=True)

    # Extract the category names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Extract the category values
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Parameters:
        df (DataFrame): A pandas dataframe including messages and a column with all categories and their values.
        database_filename (str): The name that should be used for the database.
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()