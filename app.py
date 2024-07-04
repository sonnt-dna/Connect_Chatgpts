from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import os
import io

# Initialize the FastAPI app
app = FastAPI()

# Define the request body for predictions
class PredictRequest(BaseModel):
    DEPTH: float
    CALI: float
    DT: float
    GR: float
    LLD: float
    LLS: float
    NPHI: float
    PHIE: float
    RHOB: float
    SW: float

# Load the trained model if it exists
model_path = 'linear_regression_pipeline.pkl'
columns_path = 'feature_columns.pkl'
if os.path.exists(model_path) and os.path.exists(columns_path):
    pipeline = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
else:
    pipeline = None
    feature_columns = None

# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictRequest):
    try:
        if pipeline is None or feature_columns is None:
            raise HTTPException(status_code=400, detail="Model is not trained yet. Please train the model first.")
        
        # Extract data from request
        data = np.array([[request.DEPTH, request.CALI, request.DT, request.GR, request.LLD,
                          request.LLS, request.NPHI, request.PHIE, request.RHOB, request.SW]])
        
        # Convert to DataFrame to ensure the columns match
        data_df = pd.DataFrame(data, columns=feature_columns)
        
        # Log input data
        print("Input data:", data_df)
        print("Input data shape:", data_df.shape)
        print("Feature columns used for prediction:", feature_columns)

        # Make prediction
        prediction = pipeline.predict(data_df)
        
        # Return the prediction
        return {'VWCL': prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define the training endpoint
@app.post('/train')
async def train(file: UploadFile = File(...), feature_cols: str = Form(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        data = pd.read_excel(io.BytesIO(contents))
        
        # Handle missing values by filling them with the mean of each column
        data = data.fillna(data.mean())
        
        # Define feature columns
        global feature_columns
        if feature_cols:
            feature_columns = feature_cols.split(",")
        else:
            raise HTTPException(status_code=400, detail="No feature columns provided.")
        
        # Ensure the file contains the necessary columns
        if not all(column in data.columns for column in feature_columns):
            raise HTTPException(status_code=400, detail="Input file is missing required columns.")
        
        # Features (X) and target (y)
        X = data[feature_columns]
        y = data['VWCL']
        
        # Save the feature columns
        joblib.dump(feature_columns, columns_path)
        
        # Log training data
        print("Training data shape:", X.shape)
        print("Feature columns:", feature_columns)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a column transformer to apply StandardScaler to the features only
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), feature_columns)
            ]
        )

        # Create a pipeline with the preprocessor and a linear regression model
        global pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)
        print("Predictions on test set:", y_pred)
        print("Test set shape:", X_test.shape)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the trained pipeline to a file
        joblib.dump(pipeline, model_path)
        
        return {
            'message': 'Model trained successfully',
            'Mean Squared Error': mse,
            'R-squared Score': r2
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)