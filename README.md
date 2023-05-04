# Predicting FIFA 23 Ultimate Team Player Card Values: A Machine Learning Approach

## Overview

This project is aimed at predicting the marketplace value of FIFA Ultimate Team cards based on the player's attributes. The project includes notebooks that outline the process for each stage of the project, including web scraping, data cleaning, EDA, feature selection, model building and evaluation, model analysis and a conclusion. The project also includes a pipeline for the model building which includes a python function that runs through every step. Additionally, the project includes a Flask app for model deployment, as well as a Dockerfile to containerise the model, so people can send post requests.

## Data

The data used in this project was obtained through web scraping from a popular website called FUTBIN. The data contains information about various attributes of FIFA Ultimate Team cards, such as player name, rating, position, nationality, and club.    

## Methodology

The project follows a standard machine learning pipeline, including the following stages:

- Web scraping
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature selection
- Model building and evaluation
- Model analysis and conclusion

The steps are outlined through the notebooks in the Notebooks folder where each step is explained in great detail. 

## Usage
To use this project, follow the instructions below:

1. Clone the repository to your local machine.

2. Install the required Python packages using pip install -r requirements.txt.

3. Run the pipeline using the fifa_pipeline.py module in the ModelBuilding folder in the Pipelines directory.

4. Use the flask_app module in the ModelDeployment folder in the Pipelines directory to deploy the model locally and make predictions.


Alternatively you can use the deploy.dockerfile file to deploy it. Once the model is deployed locally you can simply send a post request containing the player name and the page number the card is located on Futbin.com to return the card's marketplace value and the predicted value for the card based on the model. 


## Conclusion
In conclusion, this project demonstrates the use of machine learning techniques to predict the marketplace value of FIFA Ultimate Team cards based on player attributes. The pipeline is containerised using Docker, making it easy to reproduce and run the pipeline on different systems. 
