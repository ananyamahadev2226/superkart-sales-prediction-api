# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
import sys # For manipulating system modules

# MRP_BIN_EDGES must be defined before engineer_features if it's used within
mrp_quantiles = np.array([126.2225, 146.585, 167.505]) # Copied from notebook cell 'fet3-MuLqbBL'
MRP_BIN_EDGES = [-np.inf, mrp_quantiles[0], mrp_quantiles[1], mrp_quantiles[2], np.inf]

def feature_engineering(dataset, mrp_bin_edges):
  # 1. Store_Age
  # Assuming the current year is 2026 for calculating store age
  dataset['Store_Age'] = 2026 - dataset['Store_Establishment_Year'] #current year as 2026

  # 2. MRP_Tier
  # Binning Product_MRP into 4 tiers using quantiles (equal frequency bins)
  dataset['MRP_Tier'] = pd.cut(dataset['Product_MRP'],
                               bins=mrp_bin_edges,labels=['Low', 'Medium', 'High', 'Premium'],
                               include_lowest=True)

  # 3. Shelf_Space_Absolute
  # First, map Store_Size to numerical values
  store_size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
  dataset['Store_Size_Numeric'] = dataset['Store_Size'].map(store_size_mapping)

  # Then, create the interaction feature
  dataset['Shelf_Space_Absolute'] = dataset['Product_Allocated_Area'] * dataset['Store_Size_Numeric']

  # Drop the original columns after feature engineering
  columns_to_drop = ['Store_Establishment_Year', 'Product_MRP', 'Store_Size', 'Store_Size_Numeric']
  dataset = dataset.drop(columns=columns_to_drop, errors='ignore')

  return dataset
# --- End of Feature Engineering components ---

# Initialize the Flask application
product_store_sales_predictor_api = Flask("SuperKart Product Store Sales Predictor")

# Load the trained machine learning model

#MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_sales_model_v1_0.joblib")

# Some models expect the custom feature function under __main__.
setattr(sys.modules['__main__'], 'feature_engineering', feature_engineering)
model = joblib.load("superkart_sales_model_v1_0.joblib")

# Define a route for the home page (GET request)
@product_store_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Product Store Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@product_store_sales_predictor_api.post('/v1/sales')
def product_store_sales():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted product sales as a JSON response.
    """
    try:
        # Get the JSON data from the request body
        product_data = request.get_json()

        if not product_data:
            return jsonify({'error': 'Invalid JSON or empty request body'}), 400

        # Extract relevant features from the JSON data
        input_data = {
            'Product_Id': product_data['Product_Id'],
            'Product_Weight': product_data['Product_Weight'],
            'Product_Sugar_Content': product_data['Product_Sugar_Content'],
            'Product_Allocated_Area': product_data['Product_Allocated_Area'],
            'Product_Type': product_data['Product_Type'],
            'Product_MRP': product_data['Product_MRP'],
            'Store_Id': product_data['Store_Id'],
            'Store_Establishment_Year': product_data['Store_Establishment_Year'],
            'Store_Size': product_data['Store_Size'],
            'Store_Location_City_Type': product_data['Store_Location_City_Type'],
            'Store_Type': product_data['Store_Type']
        }

        # Convert the extracted data into a Pandas DataFrame
        input_data_df = pd.DataFrame([input_data])

        # Make prediction
        predicted_product_store_sales = model.predict(input_data_df)[0]

        # Convert predicted_price to Python float
        predicted_product_store_sales = round(float(predicted_product_store_sales), 2)

        # Return the actual price
        return jsonify({'Predicted Product store sales (in dollars)': predicted_product_store_sales})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define an endpoint for batch prediction (POST request)
@product_store_sales_predictor_api.post('/v1/salesbatch')
def product_stor_sales_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing product details for multiple or single outlets,
    and returns the predicted product store sales as a dictionary in the JSON response.
    """
    try:
        # Get the uploaded CSV file from the request
        file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        input_data = pd.read_csv(file)

        # Make predictions for all products in the DataFrame
        predicted_product_store_sales = model.predict(input_data).tolist()

        # Calculate actual prices
        predicted_rounded_sales = [round(float(sales), 2) for sales in predicted_product_store_sales]

        # Create a dictionary of predictions with product IDs as keys
        product_ids = input_data['Product_Id'].tolist()  # Assuming 'id' is the product ID column
        output_dict = dict(zip(product_ids, predicted_rounded_sales))  # Use actual prices

        # Return the predictions dictionary as a JSON response
        return output_dict
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    product_store_sales_predictor_api.run(debug=True)
