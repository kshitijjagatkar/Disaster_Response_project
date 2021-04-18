import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import pickle


import nltk

nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    :param database_filepath: Takes database filepath as an input
        It unloads the data into data frame and also creates X & y variables
    :return: It returns X, y variables and target column names
    """
    #create an engine and extract data from sql
    print("\n")
    print("Loading data into DataFrame")

    conn = sqlite3.connect(database_filepath)  # connecting to database
    df = pd.read_sql('SELECT * FROM msgs_categories', conn)  #Using sql query in pandas real_sql method reads the data
    
    #preparing data
    print()
    print("preparing data...")
    #Making X and y variables
    X = df.message.values
    y = df.loc[:,'related':].values

    category_names = df.loc[:,'related':].columns  #these are tagert(y) column names

    print("returned X, y & category_names")
    
    return X,y,category_names


def tokenize(text):
    """
    :param text: It takes text as an input. Note- It does't take string dtype
        this function performs steps for text processing such as tokenization, removing stop words,
         and also perform strip the words to it's root using lemmatizer.
    :return: returns clean tokens
    """

    tokens = word_tokenize(text)  # performing tokenizing
    stop_words = stopwords.words("english")  #taken out all stop words of english langauge in this variable
    lemmatizer = WordNetLemmatizer()  # Instantiate lemmatizer

    # performing lemmatizing plus removing stop words
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    
    
    return clean_tokens


def build_model():

    """
    Performs  Machine learning pipeline. Three estimators in which
    first two vect & tfidf are transformers & last one "clf" is classifier.
    vect: performs counting of each words.
    tfidf: performs TF-IDF operations.
    clf: Classifies output variables.
    :return: Returns pipeline Intantiator
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):

    """
    This function performs the evaluation of model.
    Task such as Predicting, finding accuracy & generating classification report
    :param model: pipeline.
    :param X_test: Input test data.
    :param y_test: target test data
    :param category_names: target column names
    :return: None
    """

    print("\n")
    print("Model Returned...")
    print("\n")
    print("Predicting Model...")

    y_pred = model.predict(X_test)  # Predicting model

    accuracy = (y_pred == y_test).mean()  # finding accuracy

    print()    
    print("Accuracy:", accuracy)
    print('\n')
    
    print('\n Classification Report')  # Generating classification reports
    print(classification_report(y_test, y_pred, target_names=category_names,zero_division=0))

    return None


def save_model(model, model_filepath):

    """
    Saving the model as pickle file.
    :param model: takes model as an input
    :param model_filepath: takes file location to save model
    :return: None
    """
    
    pickle.dump(model, open(model_filepath, 'wb')) #dumping pickle file at given location.

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print("Started tokenizing...")
        print("\n")
        print("Cleaning tokens & Training Models...")
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
