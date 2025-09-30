# Health-Insurance-Prediction-using-Flask
This is a **Flask-based REST API** for predicting health insurance premiums using a **Linear Regression model** trained on insurance data.  

The project allows you to:
- **Train** the model with a dataset
- **Test** the model accuracy using Mean Squared Error (MSE)
- **Predict** premiums for new customer inputs

 ## Features
- /train → Train the model with a CSV file
- /test → Evaluate model performance
- /prediction → Predict health insurance premium from JSON input

## Project Structure
├── health_insurance.py
├── model.pkl # Saved ML model 
├── requirements.txt 
├── README.md # Project documentation
└── Health_insurance.csv 
