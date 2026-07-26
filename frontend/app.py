import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("Super Kart Product-Store Sales Prediction App")
st.write("This tool predicts the total sales for a product in a specific store based on its details.")

# Section for online prediction
st.subheader("Enter the product and store details:")

# Collect user input for property features
product_id = st.text_input("Product ID", value="FDX123")
product_weight = st.number_input("Product Weight", min_value=4.0, value=12.65, step=0.01, format="%.2f")
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.004, max_value=0.298, value=0.068, step=0.001, format="%.3f")
product_type = st.selectbox("Product Type", ['Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Snack Foods', 'Others', 'Breakfast', 'Household', 'Breads', 'Meat', 'Hard Drinks', 'Soft Drinks', 'Fruits and Vegetables', 'Starchy Foods', 'Seafood'])
product_mrp = st.number_input("Product MRP", min_value=31.0, value=147.0, step=0.1, format="%.2f")
store_id = st.text_input("Store ID", value="OUT001")
store_establishment_year = st.number_input("Store Establishment Year",value=2001)
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 2', 'Tier 1', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2', 'Departmental Store', 'Supermarket Type1', 'Food Mart'])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Id': product_id,
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Product store sales (in dollars)']
        st.success(f"Predicted Product Store Sales (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
