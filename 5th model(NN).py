import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import time
import os

start_time = time.time()

def extract_hp(engine_str):
    match = re.search(r'(\d+\.?\d*)HP', engine_str)
    return float(match.group(1)) if match else None

def extract_displacement(engine_str):
    match = re.search(r'(\d+\.?\d*)L', engine_str)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+\.?\d*)cc', engine_str)
    return float(match.group(1)) / 1000 if match else None

def extract_cylinders(engine_str):
    match = re.search(r'(\d+)\s*Cylinders?', engine_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'V(\d+)', engine_str)
    return int(match.group(1)) if match else None

def extract_valves(engine_str):
    match = re.search(r'(\d+)V', engine_str)
    return int(match.group(1)) if match else None

def data_preprocessor(df):
    # Extract numerical features from the 'engine' column
    df['horsepower'] = df['engine'].apply(extract_hp)
    df['displacement'] = df['engine'].apply(extract_displacement)
    df['cylinders'] = df['engine'].apply(extract_cylinders)
    df['valves'] = df['engine'].apply(extract_valves)

    # Replace the symbol with NaN
    df.replace({'–': np.nan}, inplace=True)  # Updated replacement symbol

    # Drop columns with more than 25% missing values
    df_null = df.columns[df.isna().mean() > .25]
    df = df.drop(df_null, axis=1)

    # Drop unneeded columns
    unneeded = ['id', 'engine', 'model']
    df = df.drop(unneeded, axis=1)

    # Handle missing values
    categorical_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns

    for col in categorical_columns:
        if df[col].isnull().any():  # Check if there are any NaN values
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)

    for col in numeric_columns:
        if df[col].dtype == 'int' or df[col].dtype == 'float':
            df[col].fillna(df[col].mean(), inplace=True)  # Replace with mean for numeric columns

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    # Check if 'clean_title' column has only one unique value and drop it if true
    if 'clean_title' in df.columns and len(df['clean_title'].unique()) == 1:
        df = df.drop('clean_title', axis=1)

    return df

def load_data(filename):
    df = pd.read_csv(filename, compression='zip')
    data = data_preprocessor(df)
    return data

def preprocess_data(data):
    X = data.drop(columns=['price'])
    y = data['price']
    return X, y

def align_columns(train, test):
    test = test.reindex(columns=train.columns, fill_value=0)
    return test

def create_model(in_dim, optimizer='adam', init='glorot_uniform', dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(64, input_dim=in_dim, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=init))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def train_neural_network(X_train, y_train):
    # Wrap the model creation function with KerasRegressor
    model = KerasRegressor(model=create_model,
        in_dim=X_train.shape[1],
        verbose=0,
        optimizer='adam',
        init='glorot_uniform',
        dropout_rate=0.0,
        epochs=100,
        batch_size=10
    )
    
    # Define the parameter grid
    param_exp = {
        'optimizer': ['adam', 'rmsprop'],
        'epochs': [50, 100],
        'batch_size': [5, 10],
        'init': ['glorot_uniform', 'normal', 'uniform'],
        'dropout_rate': [0.0, 0.1, 0.2]
    }
    
    # Use GridSearchCV to find the best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid= param_exp, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    best_model = grid_result.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
    return y_pred

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.show()

def make_predictions(model, test_data):
    predictions = model.predict(test_data)
    return predictions

def create_submission(predictions, test):
    test_data = pd.read_csv(test, compression='zip')
    test_data['price'] = predictions
    submission = test_data[['id', 'price']]
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

def main():
    filename = 'C:\\Users\\jacob\\OneDrive\\Documents\\Python\\kaggle project\\train.csv.zip'
    test_file = 'C:\\Users\\jacob\\OneDrive\\Documents\\Python\\kaggle project\\test.csv.zip'
    data = load_data(filename)
    test = load_data(test_file)
    
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    test_aligned = align_columns(X_train, test)
    
    model = train_neural_network(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)
    
    predictions = make_predictions(model, test_aligned)
    create_submission(predictions, test_file)

if __name__ == '__main__':
    main()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.5f} seconds")