# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)   # load messages dataset
    categories = pd.read_csv(categories_filepath)  # load categories dataset
    
    return messages.merge(categories, on = ['id'])   # returned merge datasets


def clean_data(df):
    
    categories = df['categories'].str.split(';',expand=True)  # create a dataframe of the 36 individual category columns
    
    row = categories.loc[0]    # select the first row of the categories dataframe
    
    # use this row to extract a list of new column names for categories.
    category_colnames = categories.loc[0].apply(lambda x : x.split('-')[0]).to_list()
    categories.columns = category_colnames   # rename the columns of `categories
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    
    # Removing ambiguous values in the cleaned dataset( eg. df['related'] column has 3 unique values [0,1,2])
    # So we are replacing with with only two(0,1) values cause rest columns has only these values.
    for col in categories.columns:
        print(categories[col].unique(),end="")
    categories.related.replace(2,1,inplace=True)
    print("\n")
    print("\n after replacing 2 by 1",categories.related.unique())
    
    #Replace categories column in df with new category columns.
    #Drop the categories column from the df dataframe since it is no longer needed.
    df.drop(columns = 'categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # Remove duplicates.
    # check number of duplicates
    print("Number of duplicates in the dataset:",(df.duplicated() == True).sum())
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    print("Number of duplicates After dropping:",(df.duplicated() == True).sum())
    
    return df 
    
def save_data(df):
    
    #Save the clean dataset into an sqlite database.
    engine3 = create_engine('sqlite:///data/disaster_response.db')
    df.to_sql('msgs_categories', engine3, if_exists="replace",index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("\n")
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print("\n")
        print('Cleaning data...')
        df = clean_data(df)
        
        print("\n")
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df)
        
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