"""
Custom functions to be used during the FIFA project

"""

__date__ = "2023-04-29"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Data Manipulation
import numpy as np
import pandas as pd

# Web Scraping
import requests
from bs4 import BeautifulSoup

# Time
from time import sleep

import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import pycountry_convert as pc

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# %% --------------------------------------------------------------------------
# Webscraping
# # -----------------------------------------------------------------------------

# Get location atrributes for a player 
def get_loc_attrs(row):
    attrs = []
    first_col = row.find_all('td')[1]
    for link in first_col.find_all('a')[1:4]:
        attr = link.get('data-original-title')
        attrs.append(attr)
    return attrs


# Get table headers for the player table
def get_table_headers(soup):
    headers = []
    for header in soup.find_all('th')[2:]: # Start from third header as first two were in a different format
        header_title = header.find('a').get('data-original-title')
        if header_title[:8] == 'Order By': # Remove 'Order By' from header
            header_title = header_title[9:]
        headers.append(header_title)

    headers.insert(0, 'Name') # Add 'Name' to start of list as this wasn't added above
    headers.extend(['Team', 'Nation', 'League']) # Add other attributes not picked up above
    
    return headers


# Create a dataframe for all players from one page
def players_df_from_onepage(soup):
    # Create an empty list to store all the player attributes
    all_players = []
    
    # Loop through each row in the table on the page
    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        # Check if there are at least 2 columns (to skip blank lines or ads)
        if len(cols) > 1:
            # Extract the location attributes for the player from the row
            loc_attrs = get_loc_attrs(row)
            # Extract the other player attributes from the columns in the row
            other_attrs = [col.text.strip() for col in cols][1:] # Ignore first blank column
            # Combine and add to empty list
            player_attrs = other_attrs + loc_attrs
            all_players.append(player_attrs)

    # Get the table headers from the page
    headers = get_table_headers(soup)

    # Convert the list of player attributes to a pandas dataframe
    df = pd.DataFrame(all_players, columns=headers)
    
    # Rename the 'Attack \ Defense' column to 'Work Rate (Attack \ Defense)'
    df.rename(columns={'Attack \ Defense' : 'Work Rate (Attack \ Defense)'}, inplace=True)
    
    # Return the dataframe containing all the player attributes
    return df



# Create a dataframe from any page of players
def player_df_from_anypage(page_number):
    # Define url and user-agent
    url = 'https://www.futbin.com/players?page=' + str(page_number)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'

    # Set the user agent string in the headers
    headers = {
        'User-Agent': user_agent
    }

    # Make the request using the headers
    response = requests.get(url, headers=headers)

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Call the previously defined function to create a DataFrame
    df = players_df_from_onepage(soup)

    # Return the dataframe containing all the player attributes
    return df



# Final function to create player dataset
def get_dataset():
    
    # Define url and user-agent
    url = 'https://www.futbin.com/players?page=1'
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    # Set the user agent string in the headers
    headers = {
        'User-Agent': user_agent
    }
    # Make the request using the headers
    response = requests.get(url, headers=headers)
    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get the last page number
    last_page = int(soup.find_all('li', class_='page-item')[-2].find('a').text) 

    try:
        # create an empty pandas DataFrame to hold all the scraped data
        all_players = pd.DataFrame()
        
        # Loop through all pages to get all player data for that page and add to above dataframe 
        for i in range(1,last_page+1):
            df_i = player_df_from_anypage(i)
            all_players = pd.concat([all_players, df_i])
            
            # Sleep for 15 seconds every six requests to avoid too many requests
            if (i) % 6 == 0:
                sleep(15)        
            if i % 50 == 0:
                print(f'Page {i} scraped, {all_players.shape[0]} rows added')
    
    # If an error occurs, sleep for 30 seconds before continuing
    except requests.exceptions.RequestException as err:
            print(err)
            sleep(30)
    print('All pages scraped')

    return all_players

# %% --------------------------------------------------------------------------
# Data Cleaning
# -----------------------------------------------------------------------------

# Get a list of unique positions
def get_unique_positions(df):
    
    # Code to change '/n' to commas
    df['Position'] = df['Position'].apply(lambda x: ','.join(x.split()))

    # Code to extract a list of all unique positions
    position_list = df['Position'].value_counts().index.values
    nested_positions = [value.split(',') for value in position_list]
    unique_positions = list(set([pos for sublist in nested_positions for pos in sublist]))

    return unique_positions 


# Function to tidy position column
def tidy_position(df):

    # Code to change '/n' to commas
    df['Position'] = df['Position'].apply(lambda x: ','.join(x.split()))

    # Get unique positions
    unique_positions = get_unique_positions(df)

    # Add additional columns for each position
    for position in unique_positions:
        df[position] = 0

    # Populate the additional columns with ones if the player plays in that position
    for i, entry in enumerate(df['Position']):
        positions = entry.split(',')
        for pos in positions:
            df.loc[i, pos] = 1

    # Drop original column
    df.drop('Position', axis=1, inplace=True)   

    return df 


# Function to tidy Version column
def tidy_version(df):

    # Make a copy
    df1 = df.copy()

    # Remove '\n' from the Version column and split the two values into new columns
    df1['Card Type'] = df1['Version'].apply(lambda x: ' '.join(x.split()[0:-1]))
    df1['Acceleration Type'] = df1['Version'].apply(lambda x: x.split()[-1])
    df1.drop('Version', axis=1, inplace=True)

    return df1


# Funcction to convert string to float
def convert_string_to_number(value):
    if value[-1] == 'M':
        return float(value[:-1]) * 1000000
    elif value[-1] == 'K':
        return float(value[:-1]) * 1000
    else:
        return float(value)


# Function to tidy Price column
def tidy_price(df):

    # Remove '\n' and extract price
    df['Price'] = df['Price'].apply(lambda x: x.split()[0])

    # Apply conversion function on Price column
    df['Price'] = df['Price'].apply(convert_string_to_number)

    return df


# Function to tidy Work Rate column
def tidy_work_rate(df):

    # Create the new columns and drop the original
    df['Work Rate (Attack)'] = df['Work Rate (Attack \ Defense)'].str.split().str[0]
    df['Work Rate (Defense)'] = df['Work Rate (Attack \ Defense)'].str.split().str[2]
    df.drop('Work Rate (Attack \ Defense)', axis=1, inplace=True)

    return df


# Function to create Height and Weight columns
def create_height_and_weight(df):

    # Extract height and create column
    df['Height (cm)'] = df['Height'].str.split().str[0].str[:-2]
    df['Height (cm)'] = df['Height (cm)'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
    df['Height (cm)'] = df['Height (cm)'].fillna(df['Height (cm)'].mean())

    # Extract weight and create column
    df['Weight (kg)'] = df['Height'].str.split().str[4].str[1:-3]
    df['Weight (kg)'] = df['Weight (kg)'].apply(lambda x: int(x) if pd.notnull(x) and x != '' else np.nan)
    df['Weight (kg)'] = df['Weight (kg)'].fillna(df['Weight (kg)'].mean())

    # Drop orginal column
    df.drop('Height', axis=1, inplace=True)

    return df


# Function to convert dataframe to lowercase
def convert_df_lowercase(df):
    df = df.applymap(lambda x: x.lower() if type(x)==str else x)
    return df


# Function to remove players with zero value
def remove_zero_valued_players(df):
    df = df[df['Price'] != 0]
    return df


# Function to combine all data cleaning steps
def clean_dataset(df):
    df1 = df.copy()
    df2 = df1.reset_index(drop=True)
    df3 = tidy_position(df2)
    df4 = tidy_version(df3)
    df5 = tidy_price(df4)
    df6 = tidy_work_rate(df5)
    df7 = create_height_and_weight(df6)
    df8 = convert_df_lowercase(df7)
    df9 = remove_zero_valued_players(df8)
    df10 = df9.drop('Name', axis=1)
    return df10

# %% --------------------------------------------------------------------------
# Exploratory Data Analysis
# -----------------------------------------------------------------------------

# Function to fill empty Card Type values with 'normal'
def fill_empty_card_type(df):
    df1 = df.copy()
    df1['Card Type'] = df1['Card Type'].fillna('normal')
    return df1

# Reduce dataset by removing lower rated and non-special players
def remove_low_non_special(df):
    df = df[(df['Rating'] >= 90) | ~(df['Card Type'].isin(['normal', 'non-rare', 'rare', 'libertadores','sudamericana']))]
    return df

# Function to create and encode work rate columns
def encode_work_rate(df_train, df_test):

    # Create encoder object
    encoder = OrdinalEncoder(categories=[['l', 'm', 'h']], handle_unknown='use_encoded_value', unknown_value=4)

    # Fit and transorm on train data and transform test data - Attack
    df_train['WR_Att_OE'] = encoder.fit_transform(df_train[['Work Rate (Attack)']])
    df_test['WR_Att_OE'] = encoder.fit_transform(df_test[['Work Rate (Attack)']])

    # Fit and transorm on train data and transform test data - Defense
    df_train['WR_Def_OE'] = encoder.fit_transform(df_train[['Work Rate (Defense)']])
    df_test['WR_Def_OE'] = encoder.fit_transform(df_test[['Work Rate (Defense)']])

    # Drop original columns
    df_train.drop(['Work Rate (Attack)', 'Work Rate (Defense)'], axis=1, inplace=True)
    df_test.drop(['Work Rate (Attack)', 'Work Rate (Defense)'], axis=1, inplace=True)

    return df_train, df_test

# Function to OHE columns and add to dataframe
def add_ohe_columns(df_train, df_test, column):
    
    df_train2 = df_train.copy()
    df_test2 = df_test.copy()


    # Create OHE object
    ohe = OneHotEncoder(handle_unknown='ignore')

    # Fit and transform train data
    ohe.fit(df_train2[[column]])
    ohe_cols_train = ohe.transform(df_train2[[column]])
    ohe_df_train = pd.DataFrame(ohe_cols_train.toarray(), columns=ohe.get_feature_names_out())

    # Concatenate with orgiginal dataframe and drop old column
    new_df_train = pd.concat([df_train2.reset_index(drop=True), ohe_df_train], axis=1)
    new_df_train.drop(column, axis=1, inplace=True)

    # Transform test data
    ohe_cols_test = ohe.transform(df_test2[[column]])
    ohe_df_test = pd.DataFrame(ohe_cols_test.toarray(), columns=ohe.get_feature_names_out())

    # Concatenate with orgiginal dataframe and drop old column
    new_df_test = pd.concat([df_test2.reset_index(drop=True), ohe_df_test], axis=1)
    new_df_test.drop(column, axis=1, inplace=True)

    # Return new dataframe
    return new_df_train, new_df_test


# Function to replace values in a column containing parts of a string to a new value
def replace_cat_values(df_train, df_test, column, old_values, new_value):
    
    # Create copies of the dataframes 
    new_df_train = df_train.copy()
    new_df_test = df_test.copy()

    # Convert old_values to a string, if it is a list, for use in str.contains()
    if type(old_values) == str:
        string = old_values
    elif type(old_values) == list:
        string = '|'.join(old_values)
        
    # Create a mask of rows where the column contains any of the old values
    train_mask = new_df_train[column].str.contains(string)
    test_mask = new_df_test[column].str.contains(string)

    # Replace the old values with the new value in the selected rows
    new_df_train.loc[train_mask, column] = new_value
    new_df_test.loc[test_mask, column] = new_value

    # Return new df
    return new_df_train, new_df_test


# Function to group Card Type values
def group_card_types(df_train, df_test):
    # SBC (We use 'sb' as one value is missing the 'c')
    df_train2, df_test2 = replace_cat_values(df_train, df_test, 'Card Type', 'sb', 'sbc')
    # MOTM
    df_train3, df_test3 = replace_cat_values(df_train2, df_test2, 'Card Type', 'motm', 'motm')
    # Icon
    df_train4, df_test4 = replace_cat_values(df_train3, df_test3, 'Card Type', 'icon', 'icon')
    # Hero
    df_train5, df_test5 = replace_cat_values(df_train4, df_test4, 'Card Type', 'hero', 'hero')
    # Token
    df_train6, df_test6 = replace_cat_values(df_train5, df_test5, 'Card Type', 'token', 'token')
    # IF
    df_train7, df_test7 = replace_cat_values(df_train6, df_test6, 'Card Type', 'if', 'if')
    # World Cup
    df_train8, df_test8 = replace_cat_values(df_train7, df_test7, 'Card Type', 'world cup', 'world cup')
    # UEFA Cup competitions
    df_train9, df_test9 = replace_cat_values(df_train8, df_test8, 'Card Type', ['ucl','uel','uecl','champions league','europa league','conference league'], 'uefa cup comps')
    # South America competitions
    df_train10, df_test10 = replace_cat_values(df_train9, df_test9, 'Card Type', ['sudamericana','libertadores'], 'south america comps')
    # Normal cards
    df_train11, df_test11 = replace_cat_values(df_train10, df_test10, 'Card Type', ['normal','rare','non-rare'], 'non-special')
    
    # Other cards:

    # Get a list of card types not already grouped
    all_types = df_train11['Card Type'].value_counts().index.to_list()
    replaced_types = ['icon', 'south america comps', 'sbc', 'uefa cup comps', 'world cup', 'hero', 'motm', 'if']
    to_replace_types = [type for type in all_types if type not in replaced_types]

    # Replace these with 'other'
    df_train12, df_test12 = replace_cat_values(df_train11, df_test11, 'Card Type', to_replace_types, 'other') 

    return df_train12, df_test12


# Function to group League values
def group_league(df_train, df_test):

    # Get a list of league names
    league_names = df_train['League'].value_counts().index

    # Big 5 league's lower divisions:
    big5_low_divs = [league[:league.find('(')] for league in league_names if league[-2] in ['2','3','4','5']]
    df_train, df_test = replace_cat_values(df_train, df_test, 'League', big5_low_divs, 'big5_low_div')

    # South American Divisions:
    df_train, df_test = replace_cat_values(df_train, df_test, 'League', ['conmebol libertadores', 'liga dimayor ii' 'conmebol sudamericana', 'primera división (arg 1)'], 'sa_leagues')

    # Asian and Australian Divisions:
    df_train, df_test = replace_cat_values(df_train, df_test, 'League', ['a-league (aus 1)', 'indian super league (ind 1)', 'chinese fa super l. (chn 1)','k league 1 (kor 1)'], 'asia_aus_leagues')

    # Middle Eastern Leagues:
    df_train, df_test = replace_cat_values(df_train, df_test, 'League', ['united emirates l. (uae 1)', 'mbs pro league (sau 1)'], 'me_leagues')

    # Remaining European Leagues:
    rem_eur_divs = [league[:league.find('(')] for league in league_names if league[-2] in ['1','l']]
    df_train, df_test = replace_cat_values(df_train, df_test, 'League', rem_eur_divs, 'rem_eur_div')

    return df_train, df_test


# Function to get the continent from a country
def country_to_continent(country_name):
    try:
        # Convert the country name to its corresponding ISO alpha-2 code
        country_alpha2 = pc.country_name_to_country_alpha2(country_name.title())
        # Use the alpha-2 code to determine the continent code
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        # Convert the continent code to its corresponding continent name
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except LookupError:
        # If the country name is not found in the pycountry database, return "Unknown"
        return country_name


# Function to replace countries with continent
def group_country(df_train, df_test):

    # We create a new column callled 'Continent' usinf the above function
    df_train['Continent'] = df_train['Nation'].apply(lambda x: country_to_continent(x))
    df_test['Continent'] = df_test['Nation'].apply(lambda x: country_to_continent(x))

    # Replace remaining countries:
    # Europe:
    df_train, df_test = replace_cat_values(df_train, df_test, 'Continent',
                            ['england', 'scotland', 'northern ireland','republic of ireland', 'wales', 'bosnia and herzegovina','kosovo', 'fyr macedonia'],
                            'Europe')
    # Africa:
    df_train, df_test = replace_cat_values(df_train, df_test, 'Continent', ["côte d'ivoire",'congo dr', 'cape verde islands'], 'Africa')
    # Asia:
    df_train, df_test = replace_cat_values(df_train, df_test, 'Continent', ['korea republic','china pr', 'chinese taipei'], 'Asia')
    # North America:
    df_train, df_test = replace_cat_values(df_train, df_test, 'Continent', ['st. kitts and nevis', 'trinidad and tobago' ,'antigua and barbuda'], 'North America')

    # Drop Nation columns:
    df_train.drop('Nation', axis=1, inplace=True)
    df_test.drop('Nation', axis=1, inplace=True)
    
    return df_train, df_test


def group_team(df_train, df_test):

    # Return an updated list of League values
    team_names = df_train['Team'].value_counts().index

    # We can retrieve a list of the 20 teams with the most players
    top_teams = df_train['Team'].value_counts().head(20).index.to_list()

    # We can then use this to get a list of the other teams
    other_teams = [team for team in team_names if team not in top_teams]

    # Replace the other teams with 'Other'
    df_train['Team'] = df_train['Team'].replace({team : 'Other' for team in other_teams})
    df_test['Team'] = df_test['Team'].replace({team : 'Other' for team in other_teams})

    return df_train, df_test

# Function to replace zero weight values with the mean
def replace_zero_weight_value(df_train, df_test):
    df_train['Weight (kg)'] = df_train['Weight (kg)'].replace({0: df_train['Weight (kg)'].mean()})
    df_test['Weight (kg)'] = df_test['Weight (kg)'].replace({0: df_test['Weight (kg)'].mean()})

    return df_train, df_test


# Function to group Popularity into bins
def group_popularity(df_train, df_test):
    
    # Define the bin edges
    bin_edges = [-float('inf'), -1000, -100, -10, 0, 10, 100, 1000, float('inf')]

    # Bin the 'Popularity' column using the cut method
    df_train['Popularity'] = pd.cut(df_train['Popularity'], bins=bin_edges, labels=[1,2,3,4,5,6,7,8])
    df_test['Popularity'] = pd.cut(df_test['Popularity'], bins=bin_edges, labels=[1,2,3,4,5,6,7,8])

    # Convert to integers
    df_train['Popularity'] = df_train['Popularity'].astype('int')
    df_test['Popularity'] = df_test['Popularity'].astype('int')

    return df_train, df_test


# Function to get a dataframe of numerical values
def get_num_df(df_train, df_test):

    # Get a list of unique positions
    unique_positions = ['CF', 'GK',
       'CM', 'RWB', 'CDM', 'CAM', 'LM', 'RW', 'RM', 'LWB', 'RB', 'LW', 'ST',
       'CB', 'LB']

    # Get a list of encoded columns
    encoded_cols = [col for col in df_train.columns if '_' in col]

    # We can create a new dataframe that drops our encoded features
    num_df_train = df_train.select_dtypes(exclude='bool').drop(encoded_cols + unique_positions, axis=1)
    num_df_test = df_test.select_dtypes(exclude='bool').drop(encoded_cols + unique_positions, axis=1)

    return num_df_train, num_df_test


# Function to transform skewed columns
def transform_skewed_cols(df_train, df_test, skew_limit):
    
    # Get dataframes of numeric features
    num_df_train, num_df_test = get_num_df(df_train, df_test)

    # We can get the skew values for each feature
    skew_vals = num_df_train.skew().sort_values(ascending=False)

    # We get a list of all the columns with a skew value over this limit
    skew_cols = skew_vals[abs(skew_vals) > skew_limit].index.to_list()

    # Box-Cox transform the features:
    for feature in skew_cols:
        if feature == 'Price':
            continue

        # apply Box-Cox transformation to training set
        transformed_train, lam = boxcox(num_df_train[feature])
        num_df_train[feature] = pd.Series(transformed_train)
        
        # apply same transformation to test set
        transformed_test = boxcox(num_df_test[feature], lmbda=lam)
        num_df_test[feature] = pd.Series(transformed_test)

    # Get a list of numerical column names
    num_cols = num_df_train.columns.to_list()

    # Replace columns in orginal dataframes with updated columns
    df_train[num_cols] = num_df_train
    df_test[num_cols] = num_df_test

    return df_train, df_test


# Combine all EDA steps
def combine_eda_steps(df):

    # Create copy
    df1 = df.copy()

    # Fill NaN and remove low-rated, non-special cards
    df2 = fill_empty_card_type(df1)

    df3 = remove_low_non_special(df2)

    # Train/Test split
    df_train, df_test = train_test_split(
        df3, test_size=0.2, random_state=42
    )

    # Encode Work Rate
    df_train2, df_test2 = encode_work_rate(df_train, df_test)

    # OHE 'Acceleration Type
    df_train3, df_test3 = add_ohe_columns(df_train2, df_test2, 'Acceleration Type')

    # Group and OHE Card Types
    df_train4, df_test4 = group_card_types(df_train3, df_test3)
    df_train5, df_test5 = add_ohe_columns(df_train4, df_test4, 'Card Type')

    # Group and OHE League
    df_train6, df_test6 = group_league(df_train5, df_test5)
    df_train7, df_test7 = add_ohe_columns(df_train6, df_test6, 'League')

    # Group and OHE Nation
    df_train8, df_test8 = group_country(df_train7, df_test7)
    df_train9, df_test9 = add_ohe_columns(df_train8, df_test8, 'Continent')

    # Group and OHE Team
    df_train10, df_test10 = group_team(df_train9, df_test9)
    df_train11, df_test11 = add_ohe_columns(df_train10, df_test10, 'Team')

    # Replace zero weight values
    df_train12, df_test12 = replace_zero_weight_value(df_train11, df_test11)

    # Group Popularity into bins
    df_train13, df_test13 = group_popularity(df_train12, df_test12)

    # Transform skewed columns
    df_train14, df_test14 = transform_skewed_cols(df_train13, df_test13, 0.75)

    return df_train14, df_test14


# %% --------------------------------------------------------------------------
# Feature Selection
# -----------------------------------------------------------------------------

# Function to remove features based on a correlation and colinearity limit 
def drop_features(df_train, df_test, corr_limit, colin_limit):

    # View correlation values
    top_corr_vals = abs(df_train.corr()['Price']).sort_values(ascending=False)

    # Get a list of features with a correlation over corr_limit
    corr_over_lim = top_corr_vals[top_corr_vals > corr_limit].index.to_list()

    # Reduce features of df_train and df_test
    df_train2 = df_train[corr_over_lim]
    df_test2 = df_test[corr_over_lim]

    # Create a correlation matrix
    corr_matrix = df_train2.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > colin_limit)]

    # Drop the highly correlated features
    df_train3 = df_train2.drop(to_drop, axis=1)
    df_test3 = df_test2.drop(to_drop, axis=1)

    return df_train3, df_test3

# %% --------------------------------------------------------------------------
# Model Building and Evaluation
# -----------------------------------------------------------------------------

# Function to define X and y
def def_X_and_y(df_train, df_test):
    X_train = df_train.drop(['Price'], axis=1)
    X_test = df_test.drop(['Price'], axis=1)

    y_train = df_train['Price']
    y_test = df_test['Price']

    return X_train, X_test, y_train, y_test





