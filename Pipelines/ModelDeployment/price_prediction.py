"""
This is a module for performing a FIFA Ultimate Time player card price prediction.
The main functions will take the json input from the POST request containing a page
number and a player name and return the predicted valuation.
"""

__date__ = "2023-05-03"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
import logging
import json
import pickle
import fifa_functions as ff


# %% --------------------------------------------------------------------------
# Set up logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# %% --------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
try:
    with open(r'resources/xgbmodel.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    with open(r'Pipelines\ModelDeployment\resources\xgbmodel.pkl', 'rb') as f:
        model = pickle.load(f)

# %% --------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
#dataset_path = 

# %% --------------------------------------------------------------------------
# Define predict_price function
# -----------------------------------------------------------------------------
def predict_price(json_data):
    '''
    This is the main function that takes the json_data, retrievd the player info from FUTBIN,
    cleans teh data to make it in a format the model will accept, predicts the players' price, 
    and returns the result as json.
    '''
    logger.info('Trying to run predict_price')

    input_page = json_data['FUTBIN Page']
    input_name = json_data['Player Name']

    # Create a dataframe of the input page
    logger.info('Webscraping page data')
    page_df = ff.player_df_from_anypage(input_page)
    # Extract player entry
    player_entry = page_df[page_df['Name']==input_name]
    # Save to file 
    try:
        player_entry.to_csv(f'resources/searched_players/{input_name}.csv', index=False)
    except:
        player_entry.to_csv(f'Pipelines/ModelDeployment/resources/searched_players/{input_name}.csv', index=False)
    # Read file
    try: 
        player_entry = pd.read_csv(f'resources/searched_players/{input_name}.csv')
    except:
        player_entry = pd.read_csv(f'Pipelines/ModelDeployment/resources/searched_players/{input_name}.csv')
    # Clean player entry
    logger.info('Cleaning player data')
    player_entry = ff.clean_dataset(player_entry)
    # Combine with whole dataframe
    logger.info('Combining player data with original dataframe')
    try:
        original_df = pd.read_csv('resources/player_dataset_clean.csv')
    except:
        original_df = pd.read_csv('Pipelines/ModelDeployment/resources/player_dataset_clean.csv')
    combined_df = pd.concat([player_entry, original_df])
    # Replace NaN Position values with 0
    unique_positions = ['RB', 'CAM', 'CM', 'CDM', 'LWB', 'CF', 'RM', 'LW', 'LB', 'CB', 'RWB', 'ST', 'GK', 'RW', 'LM']
    combined_df[unique_positions] = combined_df[unique_positions].fillna(0)
    # Reset index and add a column for new index
    combined_df = combined_df.reset_index(drop=True)
    combined_df['orginal_index'] = combined_df.index
    # Remove duplicated value if player already exists in dataframe
    combined_df = combined_df.drop_duplicates()
    # Apply EDA steps on dataframe
    logger.info('Applying EDA steps')
    combined_df_eda = ff.combine_eda_steps(combined_df)
    # Retrieve orginal player entry
    logger.info('Retrieving original player entry')
    if not (combined_df_eda[1][combined_df_eda[1]['orginal_index']==0]).empty:
        player_entry = combined_df_eda[1][combined_df_eda[1]['orginal_index']==0]
    else:
        player_entry = combined_df_eda[0][combined_df_eda[0]['orginal_index']==0]
    # Extract market price of player
    market_price = player_entry['Price'].values[0]  
    # Generate a list of features used in model
    logger.info('Retrieving list of feature names from model')
    features = model.get_booster().feature_names
    # Extract these features from the player entry
    player_entry = player_entry.reindex(features, axis=1)
    # Fill any possible NaN values with 0
    player_entry = player_entry.fillna(0)
    # Predict value using model
    logger.info('Predicting player price using model')
    predicted_price = model.predict(player_entry)[0]

    # Now we have both the market and predicted price, 
    # we can add to a dictionary and convert to json format
    
    # Convert the float32 objects to Python floats
    predicted_price = float(predicted_price)
    market_price = float(market_price)
    
    # Create the results dictionary
    logger.info('Creating results dictionary')
    results_dict = {'Player Name': input_name,
                    'Predicted Price': predicted_price, 
                    'Marketplace Price': market_price}
    
    # Save results
    results_df = pd.DataFrame(results_dict, index=[0])
    try:
        results_df.to_csv(f'resources/searched_players/{input_name}.csv', index=False)
    except:
        results_df.to_csv(f'Pipelines/ModelDeployment/resources/searched_players/{input_name}.csv', index=False)
    return json.dumps(results_dict)











# %%
