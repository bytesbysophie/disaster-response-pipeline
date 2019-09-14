# Disaster Response Pipeline Project

## Table of Contents
1. [Installation](#Installation)
2. [Project Motivation and Description](#Project-Motivation)
3. [File Descriptions](#File-Descriptions)
4. [Authors and Acknowledgements](#Authors-Acknowledgements)

## Installation <a name="Installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (alternatively: http://localhost:3001/) to access the web app, that should look like this:
<br><br>
<img src="https://raw.githubusercontent.com/bytesbysophie/disaster-response-pipeline/master/app/app_screenshot.PNG"/>

## Project Motivation and Description <a name="Project-Motivation"></a>
The goal of this project is to create a web app where an emergency worker can input messages and receive a machine learning based classification of the message in terms of what kind of help is needed.

The key parts of the project are:
* An ETL pipeline that loads the provided and labeld message data and prepares it be be used as input for a machine learning model.
* A ML (machine learning) pipeline that builds and trains and saves a multi output classification model.
* A web app that allows users to classify new messages using the saved machine learing model.

## File Descriptions <a name="File-Descriptions"></a>

* app/
    * [run.py](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/app/run.py): Code to start the web app
    * templates/
        * [go.html](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/app/templates/go.html): HTML template for the message classification result page
        * [master.html](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/app/templates/master.html): HTML template for the start page of the web app
* data/
    * [disaster_categories.csv](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/data/disaster_categories.csv): Message category data used to train the message classifier
    * [disaster_messages.csv](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/data/disaster_messages.csv): Message data used to train the message classifier
    * [process_data.py](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/data/process_data.py): ETL pipeline to load, transform and save the training data
* models/
    * [train_classifier.py](https://github.com/bytesbysophie/disaster-response-pipeline/blob/master/models/train_classifier.py): ML pipeline that loads data and builds, evaluates and saves the classifier


## Authors and Acknowledgements <a name="Authors-Acknowledgements"></a>
This project has been implemented as part of the Udacity Data Scientist Nanodegree program. All data as well as project structure and templates for the scripts have been provided by Udacity.
