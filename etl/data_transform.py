#pip install -r requirements.txt
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def handle_typo(data):
    data['BATH'] = data['BATH'].apply(lambda x: 2 if 2 < x < 3 else x)
    data['TYPE'] = data['TYPE'].replace({'Condop for sale': 'Condo for sale'})
    return data

def handle_sublocality(data):
    # Define the replacement dictionary
    replacement_dict = {
        'Bronx County': 'The Bronx',
        'East Bronx': 'The Bronx',
        'Riverdale': 'The Bronx',
        'Kings County': 'Brooklyn',
        'Coney Island': 'Brooklyn',
        'Brooklyn Heights': 'Brooklyn',
        'Snyder Avenue': 'Brooklyn',
        'Fort Hamilton': 'Brooklyn',
        'Dumbo': 'Brooklyn',
        'New York County': 'Manhattan',
        'New York': 'Manhattan',
        'Richmond County': 'Staten Island',
        'Queens County': 'Queens',
        'Jackson Heights': 'Queens',
        'Flushing': 'Queens',
        'Rego Park': 'Queens'
    }
    # Replace values in 'SUBLOCALITY' column
    data['SUBLOCALITY'].replace(replacement_dict, inplace=True)

    return data

def handle_house_type(data):
    # Step 1: Replace specific house types with 'unspecified'
    data.loc[data['TYPE'].isin(['Foreclosure', 'Contingent', 'Coming Soon', 'Pending', 'For sale']), 'TYPE'] = "unspecified"

    # Step 2: Label encode 'SUBLOCALITY'
    label_encoder = LabelEncoder()
    data['SUBLOCALITY_label_encoded'] = label_encoder.fit_transform(data['SUBLOCALITY'])

    # Step 3: Separate known and unspecified type data
    known_type_data = data[data['TYPE'] != 'unspecified']
    unspecified_type_data = data[data['TYPE'] == 'unspecified']

    # Step 4: Define features (X) and target (Y)
    X = known_type_data[['SUBLOCALITY_label_encoded', 'PROPERTYSQFT', 'PRICE', 'BEDS', 'BATH']]
    Y = known_type_data['TYPE'] 

    # Step 5: Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Step 6: Initialize and train the classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, Y_train)

    # Step 7: Test the model and display accuracy
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Housing type imputation accuracy: {accuracy}")
    
    # Step 8: Impute missing types
    if not unspecified_type_data.empty:
        data.loc[data['TYPE'] == 'unspecified', 'TYPE'] = clf.predict(unspecified_type_data[['SUBLOCALITY_label_encoded', 'PROPERTYSQFT', 'PRICE', 'BEDS', 'BATH']])
    
    return data

def handle_outliers(data):
    # Select numeric columns from the dataset
    numeric_columns = data.select_dtypes(include=['number']).columns
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each numeric column
    Q1_numeric = data[numeric_columns].quantile(0.25)
    Q3_numeric = data[numeric_columns].quantile(0.75)
    
    # Calculate the Interquartile Range (IQR) for each numeric column
    IQR_numeric = Q3_numeric - Q1_numeric
    
    # Define the outlier cutoff threshold (6 times the IQR)
    outlier_cutoff_numeric = 6 * IQR_numeric

    # Filter the data to exclude outliers based on the calculated cutoff
    house_data_filtered = data[~((data[numeric_columns] < (Q1_numeric - outlier_cutoff_numeric)) | 
                                 (data[numeric_columns] > (Q3_numeric + outlier_cutoff_numeric))).any(axis=1)]

    # Return the filtered data
    return house_data_filtered

def handle_zip_code(data):
    # Define a function to extract the 5-digit zip code from an address string
    def extract_zip(address):
        zip_code = re.findall(r"\b\d{5}\b", address)
        return int(zip_code[0]) if zip_code else None

    # Apply the extract_zip function to the 'FORMATTED_ADDRESS' column to create 'zip_code' column
    data['zip_code'] = data['FORMATTED_ADDRESS'].apply(extract_zip)
    
    # Fill missing zip codes with the default value of 11372
    data['zip_code'].fillna(11372, inplace=True)

    # Return the modified DataFrame
    return data

def handle_close_to_attraction(data):
    # Define the Haversine distance calculation function
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon1 - lon2
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    # Dictionary of attractions and their coordinates
    attractions = {
        'Times Square': (40.758896, -73.98513),
        'Central Park': (40.7825, -73.9661),
        'World Trade Center': (40.7127, -74.013382),
        'Wall Street': (40.706005, -74.008827),
        'Stairs to Brooklyn Bridge': (40.7007, -73.9898),
        'Brooklyn Botanic Garden': (40.6677, -73.9636),
        'Flushing Meadows Corona Park': (40.7400, -73.8407),
        'Queens Zoo': (40.7437, -73.8486),
        'Yankee Stadium': (40.829659, -73.926186),
        'Bronx Zoo': (40.850278, -73.878333),
        'Sailors Snug Harbor': (40.6425, -74.1028)
    }

    # Calculate distance to each attraction and add a column to the DataFrame
    for attraction, (lat2, lon2) in attractions.items():
        data[attraction] = data.apply(lambda row: haversine_distance(row['LATITUDE'], row['LONGITUDE'], lat2, lon2), axis=1)

    # Threshold distance for being "close" (in kilometers)
    close_threshold = 2.41402  # Approx. 25 minutes walk

    # Calculate how many attractions are within the close threshold
    central_attractions = ['Times Square', 'Central Park', 'World Trade Center', 'Wall Street', 'Stairs to Brooklyn Bridge']
    data['is close to central areas'] = data[central_attractions].apply(lambda row: (row <= close_threshold).sum(), axis=1)
    
    return data

def drop_columns(data):
    selected_columns = ['Times Square','Central Park', 'World Trade Center', 'Wall Street',
                    'Stairs to Brooklyn Bridge', 'Brooklyn Botanic Garden',
                    'Flushing Meadows Corona Park', 'Queens Zoo',
                    'Bronx Zoo', 'Yankee Stadium',
                    'Sailors Snug Harbor']
    data = data.drop(columns = selected_columns)

    return data

def main():
    # Load the data
    data = pd.read_csv('data/bronze/NY-House-Dataset.csv')

    # Apply the data preprocessing functions in sequence
    data = handle_typo(data)
    data = handle_sublocality(data)
    data = handle_house_type(data)
    data = handle_outliers(data)
    data = handle_zip_code(data)
    data = handle_close_to_attraction(data)

    # Optionally, drop the columns for attractions distances if not needed
    data = drop_columns(data)

    # Save the cleaned and processed data
    data.to_csv('data/silver/NY-House-Dataset-Cleaned.csv', index=False)

    print("Data preprocessing completed and saved to 'NY-House-Dataset-Cleaned.csv'")

# Call the main function
if __name__ == "__main__":
    main()
