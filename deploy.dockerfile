# Note: To run this dockerfile you will need to have the files xgbmode.pkl and 
# player_dataset_clean.csv in the ModelDeployment/resources folder. If you do 
# not have these then run the model building pipeline first.

FROM ubuntu

RUN apt update

RUN apt -y install python3-dev python3-pip

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

COPY Pipelines/ModelDeployment /ModelDeployment/
COPY Pipelines/ModelDeployment/resources /ModelDeployment/resources/

WORKDIR /ModelDeployment

CMD ["python3", "flask_app.py"]

# Build with: docker build -f deploy.dockerfile -t fifa_price_prediction_app_deploy .

# Run with: docker run -it -p 5050:5050 fifa_price_prediction_app_deploy
