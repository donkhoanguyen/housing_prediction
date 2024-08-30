#pip install -r requirements.txt
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor

data = pd.read_csv('data/silver/NY-House-Dataset-Cleaned.csv')

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    scaler = StandardScaler()
    data['PRICE_standardized'] = scaler.fit_transform(data[['PRICE']])
    
    encoder = LabelEncoder()
    data['encoded_sublocality'] = encoder.fit_transform(data['SUBLOCALITY'])
    data['encoded_type'] = encoder.fit_transform(data['TYPE'])
    
    X = data[['encoded_type', 'BEDS', 'BATH', 'PROPERTYSQFT', 'encoded_sublocality', 'zip_code', 'is close to central areas']]
    Y = data['PRICE']
    
    return X, Y

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': Pipeline([
            ('scaler', RobustScaler()),
            ('lr', LinearRegression())
        ]),
        'SVM': Pipeline([
            ('scaler', RobustScaler()),
            ('svm', SVR())
        ]),
        'Random Forest': Pipeline([
            ('scaler', RobustScaler()),
            ('rf', RandomForestRegressor())
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', RobustScaler()),
            ('gb', GradientBoostingRegressor())
        ]),
    }

    param_grids = {
        'Linear Regression': {'lr__fit_intercept': [True, False]},
        'SVM': {
            'svm__kernel': ['linear', 'rbf'],
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto']
        },
        'Random Forest': {
            'rf__n_estimators': [450, 500, 550, 600, 650, 700, 800],
            'rf__max_depth': [10, 13, 15, 16, 17, 18, 19, 20],
            'rf__bootstrap': [True]
        },
        'Gradient Boosting': {
            'gb__n_estimators': [175, 200, 225, 250, 275, 300, 325, 350],
            'gb__learning_rate': [0.01, 0.1, 0.2, 0.4],
            'gb__max_depth': [3, 5, 6, 7, 8]
        },
    }

    best_model = None
    best_model_name = None
    best_r2 = -float('inf')  # Initialize with negative infinity for maximization
    results = {}

    for model_name, model_pipeline in models.items():
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        y_test_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        results[model_name] = {
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = model_name

        print(f"Model: {model_name}")
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Absolute Percentage Error: {mape}")
        print(f"Best Parameters: {grid_search.best_params_}\n")
    
    if best_model:
        # Export the best model
        joblib.dump(best_model, 'best_model.pkl')
        print(f"Best Model: {best_model_name} exported as 'best_model.pkl'")

    return results

def plot_performance(results):
    models_list = list(results.keys())
    mape_scores = [results[model]['mape'] for model in models_list]
    r2_scores = [results[model]['r2'] for model in models_list]

    plt.figure(figsize=(10, 6))
    plt.scatter(models_list, mape_scores, color='blue', label='MAPE')
    plt.scatter(models_list, r2_scores, color='red', label='R-squared')
    plt.title('Performance Comparison of Different Models')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    file_path = 'data/silver/NY-House-Dataset-Cleaned.csv'
    X, Y = load_and_preprocess_data(file_path)
    
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Plot results
    plot_performance(results)

if __name__ == "__main__":
    main()