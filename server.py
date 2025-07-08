# FastAPI Server for Heart Disease Prediction
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load the trained model and scaler
try:
    model = joblib.load('heart_disease_voting_model.pkl')
    print("Model loaded successfully!")
    
    # Try to load scaler if it exists
    try:
        scaler = joblib.load('heart_disease_scaler.pkl')
        print("Scaler loaded successfully!")
    except:
        print("Warning: Scaler not found. Will use default scaling parameters.")
        scaler = None
        
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Pydantic model for API input
class HeartData(BaseModel):
    age: int
    sex: str  # 'Male' or 'Female'
    chest_pain_type: str  # 'ATA', 'NAP', 'ASY', 'TA'
    resting_bp: int
    cholesterol: float
    fasting_bs: int  # 0 or 1
    resting_ecg: str  # 'Normal', 'ST', 'LVH'
    max_hr: int
    exercise_angina: str  # 'Yes' or 'No'
    oldpeak: float
    st_slope: str  # 'Up', 'Flat', 'Down'

# Create templates directory structure
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with chatbot interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_heart_disease(data: HeartData):
    """API endpoint for heart disease prediction"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input data with proper feature engineering
        input_data = pd.DataFrame([{
            'Age': data.age,
            'Sex': data.sex.upper(),
            'ChestPainType': data.chest_pain_type.upper(),
            'RestingBP': data.resting_bp,
            'Cholesterol': data.cholesterol,
            'FastingBS': data.fasting_bs,
            'RestingECG': data.resting_ecg.upper(),
            'MaxHR': data.max_hr,
            'ExerciseAngina': data.exercise_angina.upper(),
            'Oldpeak': data.oldpeak,
            'ST_Slope': data.st_slope.upper()
        }])
        
        # Apply same preprocessing as training
        # 1. Convert categorical variables to category type
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_cols:
            input_data[col] = input_data[col].astype('category')
        
        # 2. Feature engineering (same as training)
        input_data['Age_MaxHR_Ratio'] = input_data['Age'] / input_data['MaxHR']
        input_data['BP_Stress_Diff'] = input_data['RestingBP'] - (input_data['Oldpeak'] * 10)
        
        # 3. Create age groups and cholesterol levels
        input_data['AgeGroup'] = pd.cut(input_data['Age'], bins=[0, 40, 55, 65, 100],
                                labels=['<40', '40-55', '55-65', '65+'])
        input_data['Chol_Level'] = pd.cut(input_data['Cholesterol'], bins=[0, 200, 240, 1000],
                                  labels=['Normal', 'Borderline', 'High'])
        
        # 4. Create cardiac risk score
        input_data['Cardiac_Risk_Score'] = (
            input_data['Age'] * 0.3 + 
            input_data['Cholesterol'] * 0.2 + 
            input_data['RestingBP'] * 0.2 + 
            (200 - input_data['MaxHR']) * 0.3
        )
        
        # 5. Create polynomial features
        input_data['Age_Squared'] = input_data['Age'] ** 2
        input_data['Cholesterol_Squared'] = input_data['Cholesterol'] ** 2
        
        # 6. Add missing flag for cholesterol (always 0 for new predictions)
        input_data['CholesterolMissing'] = 0
        
        # 7. Apply scaling to numerical features (same as training)
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                         'CholesterolMissing', 'Age_MaxHR_Ratio', 'BP_Stress_Diff', 
                         'Cardiac_Risk_Score', 'Age_Squared', 'Cholesterol_Squared']
        
        # Create a copy for scaling
        input_scaled = input_data.copy()
        
        # Apply scaling (using training statistics if scaler is available)
        if scaler is not None:
            # Use the saved scaler
            input_scaled[numerical_cols] = scaler.transform(input_scaled[numerical_cols])
        else:
            # Manual scaling using training statistics (as fallback)
            # These are the approximate statistics from your training data
            training_stats = {
                'Age': {'mean': 54.37, 'std': 9.08},
                'RestingBP': {'mean': 132.40, 'std': 18.51},
                'Cholesterol': {'mean': 244.88, 'std': 51.83},
                'FastingBS': {'mean': 0.23, 'std': 0.42},
                'MaxHR': {'mean': 136.81, 'std': 25.46},
                'Oldpeak': {'mean': 0.89, 'std': 1.07},
                'CholesterolMissing': {'mean': 0.19, 'std': 0.39},
                'Age_MaxHR_Ratio': {'mean': 0.42, 'std': 0.14},
                'BP_Stress_Diff': {'mean': 123.51, 'std': 18.85},
                'Cardiac_Risk_Score': {'mean': 122.27, 'std': 18.14},
                'Age_Squared': {'mean': 3045.48, 'std': 1003.59},
                'Cholesterol_Squared': {'mean': 62569.93, 'std': 26079.49}
            }
            
            for col in numerical_cols:
                if col in training_stats:
                    input_scaled[col] = (input_scaled[col] - training_stats[col]['mean']) / training_stats[col]['std']
        
        # Select top features used in training
        top_features = [
            'ST_Slope', 'ChestPainType', 'Oldpeak', 'MaxHR', 'ExerciseAngina',
            'Age_MaxHR_Ratio', 'BP_Stress_Diff', 'Cholesterol', 'Age', 'RestingBP'
        ]
        
        input_features = input_scaled[top_features]
        
        # Make prediction
        prediction_proba = model.predict_proba(input_features)[0]
        prediction = model.predict(input_features)[0]
        
        # Determine risk level
        risk_score = prediction_proba[1] * 100
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return {
            "prediction": int(prediction),
            "risk_probability": round(risk_score, 2),
            "risk_level": risk_level,
            "message": "Heart disease predicted" if prediction == 1 else "No heart disease predicted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/chat", response_class=HTMLResponse)
async def chat_predict(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    chest_pain_type: str = Form(...),
    resting_bp: int = Form(...),
    cholesterol: float = Form(...),
    fasting_bs: int = Form(...),
    resting_ecg: str = Form(...),
    max_hr: int = Form(...),
    exercise_angina: str = Form(...),
    oldpeak: float = Form(...),
    st_slope: str = Form(...)
):
    """Chat interface endpoint"""
    try:
        # Create HeartData object
        heart_data = HeartData(
            age=age, sex=sex, chest_pain_type=chest_pain_type,
            resting_bp=resting_bp, cholesterol=cholesterol, fasting_bs=fasting_bs,
            resting_ecg=resting_ecg, max_hr=max_hr, exercise_angina=exercise_angina,
            oldpeak=oldpeak, st_slope=st_slope
        )
        
        # Get prediction
        result = await predict_heart_disease(heart_data)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "form_data": heart_data.model_dump(),
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)