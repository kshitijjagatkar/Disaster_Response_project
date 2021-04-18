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
    #create an engine and extract data from sql
    print("\n")
    print("Loading data into DataFrame")

    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM msgs_categories', conn)
    
    #preparing data
    print()
    print("preparing data...")
    #Making X and y variables
    X = df.message.values
    y = df.loc[:,'related':].values

    category_names = df.loc[:,'related':].columns

    print("returned X, y & category_names")
    
    return X,y,category_names


def tokenize(text):

    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    
    
    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):

    print("\n")
    print("Model Returned...")
    print("\n")
    print("Predicting Model...")

    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).mean()

    print()    
    print("Accuracy:", accuracy)
    print('\n')
    
    print('\n Classification Report')
    print(classification_report(y_test, y_pred, target_names=category_names,zero_division=0))

    return None


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))

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