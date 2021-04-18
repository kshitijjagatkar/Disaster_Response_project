# Disaster_Response_project
Web API that classifies disaster messages.

1. Project Motivation
2. Project Components
3. File Descriptions
4. Installation
5. Results
6. Licensing, Authors, and Acknowledgements

## 1. Project Motivation
 I recently gained some data engineering skills So, I found out this project can help me build an overall intution about Data engineering.
 I applied these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## 2. Project Components
This project basically divided into three sections as follows:
1.   ETL (Extract Transform Load) Pipeline
     * Loads the messages and categories datasets
     * Merges the two datasets
     * Cleans the data
     * Stores it in a SQLite database
2.  Machine Learning Pipeline
    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file
4.  Flask Web App
    * Load the pickle file
    * Clean(Tokenize) a query 
    * Predict the model
    * Make visualization of data
    * Produce the final result
 
## 3. File Description
1. Data Directory
      * This folder contains "process_data.py" which has all ETL code and perform pipeline tasks which takes input files "message.csv" &
        "categories.csv" produces database file for our model.
2. Model Directory
      * This folder has "train_classifier.py" which takes database file perform ML pipeline and produces pickle file to use it in our flask app.
4. App directory
      * this folder contains web page code. "run.py" takes care of running server & all our code. upper folders are templetes and static which contain html 
        & css, java files   respectively.

## 4. Installation
After downloading you need to have your fav code editor, I use pycharm. It's your choice what to use. Just download necessary packages before/while running program.
then all good to go.

## 5. Results
I tried with couple of algorithms, with various parameters & also with custom estimators after examining all those things I come with this solution which outputs
better results and performs faster compare to all those methods. If you wanna see those Exploratory analysis then checkout that repo.
It performs with 94.85 accuracy I have also provided classification reports.

![Screenshot (256)](https://user-images.githubusercontent.com/33245369/115155114-1013af80-a09c-11eb-8f27-74dc4886b2a5.png)


    
