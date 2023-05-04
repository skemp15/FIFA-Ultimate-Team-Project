"""
Pipeline for websraping data, cleaning data and building 
a model for predicting player prices on FIFA 23 Ultimate Team

"""

__date__ = "2023-04-29"
__author__ = "SamKemp"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# Data manipulation 
import pandas as pd

# Helper functions
import fifa_functions as ff

# XGBoost 
from xgboost import XGBRegressor

# Hyperparameter Optimisation
import optuna

# Model Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Saving model
import pickle

# %% --------------------------------------------------------------------------
# Webscraping
# -----------------------------------------------------------------------------
print('Model building started')

# Function to create dataset
df = player_dataset = ff.get_dataset()

print('Webscraping complete')


# %% --------------------------------------------------------------------------
# Data Cleaning
# -----------------------------------------------------------------------------
# Function to clean dataset
df = ff.clean_dataset(df)

try:
    df.to_csv(r'Pipelines\ModelBuilding\datasets\player_dataset_clean.csv', index=False)
except:
    df.to_csv(r'..\ModelBuilding\resources\player_dataset_clean.csv', index=False)

# Save dataset for model deployment
try:
    df.to_csv(r'Pipelines\ModelDeployment\resources\player_dataset_clean.csv', index=False)
except:
    df.to_csv(r'..\ModelDeployment\resources\player_dataset_clean.csv', index=False)


print('Data cleaning complete')

# %% --------------------------------------------------------------------------
# Exploratory Data Analysis
# -----------------------------------------------------------------------------

try:
    df = pd.read_csv(r'Pipelines\ModelBuilding\datasets\player_dataset_clean.csv')
except:
    df = pd.read_csv(r'..\ModelBuilding\resources\player_dataset_clean.csv')

# Function to run through EDA steps
df_train, df_test = ff.combine_eda_steps(df)

print('EDA complete')

# %% --------------------------------------------------------------------------
# Feature Selection
# -----------------------------------------------------------------------------
df_train, df_test = ff.drop_features(df_train, df_test, 0.2, 0.75)

print('Feature Selection complete')

# %% --------------------------------------------------------------------------
# Model Building
# -----------------------------------------------------------------------------

X_train, X_test, y_train, y_test = ff.def_X_and_y(df_train, df_test)

# Function for XGBoost hyperparameter optimisation
def objective(trial):

    # Define hyperparameters to be optimized
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'reg:squarederror',
        'random_state': 42
    }


    # Create an XGBoost regressor model with the given hyperparameters
    model = XGBRegressor(**param)

    # Compute cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Calculate the mean R2 score across cross-validation folds
    return cv_scores.mean()

study = optuna.create_study(direction='maximize') # Optimise for maximum R2 score
study.optimize(objective, n_trials=200)

xgb = XGBRegressor(
                            n_estimators = study.best_params['n_estimators'],
                            max_depth = study.best_params['max_depth'],
                            learning_rate = study.best_params['learning_rate'],
                            subsample = study.best_params['subsample'],
                            colsample_bytree = study.best_params['colsample_bytree'],
                            min_child_weight = study.best_params['min_child_weight'],
                            random_state=42
)

# Fit the regressor with the training data
xgb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = xgb.predict(X_test)

# Evaluate the performance of the model using, mean absolute error, root mean squared error and R2 score
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Save model for model deployment
try:
    with open(r'Pipelines\ModelDeployment\resources\xgbmodel.pkl', 'wb') as f:
        pickle.dump(xgb, f)
except:
    with open(r'..\ModelDeployment\resources\xgbmodel.pkl', 'wb') as f:
        pickle.dump(xgb, f)

print('Model Building complete')

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R2 Score: {r2:.4f}')
# %%
