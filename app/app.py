# Import libraries

from flask import Flask, render_template, request

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib

from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')  # rendering Home page of web-dashboard


@app.route('/ds')
def ds():
    return render_template('ds.html')  # rendering data Science web page


############################################################################
def tokenize(text):
    """
    This function processes each message and transform into tokens.
    this process includes tokenizing, lemmatizing with conversion to lowercase & striping extra space

    :param text:  Messages that to be tokenize.
    :return: Clean token of  messages
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('msgs_categories', engine)

# load model
model = joblib.load("../models/trained_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/master')
def master():
    # extract data needed for visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ###############################
    target_labels = df.iloc[:, 4:]
    target_labels.drop(columns=['child_alone'], inplace=True)
    target_frequency = [round((target_labels[col].value_counts(1)[1] * 100), 2) for col in target_labels.columns]
    target_cols = target_labels.columns
    ###############################

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=target_cols,
                    y=target_frequency
                )
            ],

            'layout': {
                'title': 'Frequency of categories',
                'yaxis': {
                    'title': "Categories Frequency in %"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True)
