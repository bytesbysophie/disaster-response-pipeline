import sys
import nltk
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Parameter:
        database_filepath (str): The filepath to the database file.
        
    Returns:
        DataFrame: A pandas dataframe holding the messages.
        DataFrame: A pandas dataframe holding the categories.
        List: A list of strings holding the categories names.
        
    This function: 
        1. loads the messages and categories from the database, 
        2. splits the data into two dataframes holding the messages and the categories.
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]

    return X, Y, Y.columns


def tokenize(text):
    '''
    Parameter:
        text (str): The text to be tokenized.
        
    Returns:
        List: A list of clean tokens extracted from the text.
        
    This function: 
        1. replaces all urls in the text with a generic placeholder, 
        2. splits the text into tokens,
        3. performs lemmatization (grouping different forms of a word),
        4. transforms all tokens to lower case.
    '''

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''        
    Returns:
        GridSearchCV: A cross validated sklearn pipeline model.
        
    This function 
        1. builds a sklearn pipeline model,
        2. defines various model parameters,
        3. creates a cross validation model using the pipline model and parameters.
    '''
 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20, min_samples_split=3)))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [3, 5, 20],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Parameter:
        model (MultiOutputClassifier): The model to be evaluated.
        X_test (DataFrame): The independent features.
        Y_test (DataFrame): The dependent features to be predicted by the model.
        category_names (list): A list of strings holding the Y_test feature names.
        
    This function: 
        1. predicts the Y_test features based on X_test,
        2. prints a classification_report for every predicted feature.
    '''

    y_pred = model.predict(X_test)

    for i in range(y_pred.shape[1]):
        print("Results for category {}:".format(category_names[i]))
        print(classification_report(Y_test.iloc[i], y_pred[i]))
        print("*"*53)

    
def save_model(model, model_filepath):
    '''
    Parameter:
        model (MultiOutputClassifier): The model to be saved.
        model_filepath (str): The filepath where the model should be saved.
        
    This function saves the model in the model_filepath as a pickle.
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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