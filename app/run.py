import json
import plotly
import numpy
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data for first graph
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    
    # extract data for second graph
    categories_counts =  df.iloc[:,4:].sum().sort_values(ascending=True)[-10:]
    categories_names = categories_counts.index

    # extract data for third graph
    related_values = [df['related'].sum(), df.shape[0]]
    related_labels = ['related', 'not related']

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
                    y=categories_names,
                    x=categories_counts,
                    orientation='h'
                )
            ],

            'layout': {
                'title': '10 Most common Message Categories',
                'xaxis': {
                    'title': "Count"
                },
                'margin': {
                    'l': 100
                }        
            }
        },
        {
            'data': [
                Pie(
                    labels=related_labels, 
                    values=related_values,
                    hole=.2,
                    scalegroup='one'
                )
            ],

            'layout': {
                'title': 'Related vs. Unrelated Messages',
                'autosize': True,
            }
        },
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()