"""
This is the Flask app for serving the regression model for predicting
a player's price in the FIFA Ultimate Team marketplace.
"""

__date__ = "2023-05-03"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from flask import Flask, request 
import logging 
from waitress import serve
import price_prediction as pp


# %% --------------------------------------------------------------------------
# Set up logger
# -----------------------------------------------------------------------------

logging.basicConfig(
    format='[%(levelname)s %(name)s] %(asctime)s - %(message)s',
    level = logging.INFO,
    datefmt='%Y/%m/%d %I:%M:%S %p'
)

logger = logging.getLogger(__name__)

# %% --------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def hello():
    logger.info('Access to landing pge')
    """
    Landing page for FIFA Ultimate Team Price Prediction model
    """
    return('Hello this is the landing page for FIFA Ultimate Team Price Prediction model')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    logger.info('Access to price prediction')
    json_data = request.get_json()
    response = pp.predict_price(json_data)
    return response

serve(app, port=5050, host='0.0.0.0')
# %%
