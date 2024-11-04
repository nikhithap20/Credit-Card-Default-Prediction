#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gradio as gr
import pandas as pd
import joblib

# Load the saved model and preprocessing objects
model_data = joblib.load('C:/Users/palug/models/credit_default_model.joblib')
model = model_data['model']
scaler = model_data['scaler']
selected_features = model_data['features']

def predict_default(limit_bal, age, pay_0, pay_2, pay_3, 
                   bill_amt1, bill_amt2, pay_amt1, pay_amt2,
                   education, marriage):
    try:
        # Create input data dictionary
        input_dict = {
            'LIMIT_BAL': float(limit_bal),
            'AGE': float(age),
            'PAY_0': float(pay_0),
            'PAY_2': float(pay_2),
            'PAY_3': float(pay_3),
            'BILL_AMT1': float(bill_amt1),
            'BILL_AMT2': float(bill_amt2),
            'PAY_AMT1': float(pay_amt1),
            'PAY_AMT2': float(pay_amt2),
            'EDUCATION': float(education),
            'MARRIAGE': float(marriage)
        }
        
        # Create DataFrame with the same feature order as training
        input_df = pd.DataFrame([input_dict])[selected_features]
        
        # Scale numeric features
        numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Format result
        if prediction == 1:
            return f"Prediction: Client WILL default (Probability: {probability:.2%})"
        else:
            return f"Prediction: Client will NOT default (Probability: {probability:.2%})"
    
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Credit Limit Balance"),
        gr.Number(label="Age"),
        gr.Number(label="Pay Status Month 0 (-1=early, 0=on time, 1-9=delay)"),
        gr.Number(label="Pay Status Month 2"),
        gr.Number(label="Pay Status Month 3"),
        gr.Number(label="Bill Amount 1"),
        gr.Number(label="Bill Amount 2"),
        gr.Number(label="Payment Amount 1"),
        gr.Number(label="Payment Amount 2"),
        gr.Number(label="Education (1=graduate, 2=university, 3=high school, 4=others)"),
        gr.Number(label="Marriage (1=married, 2=single, 3=others)")
    ],
    outputs=gr.Text(label="Default Prediction Result"),
    title="Credit Card Default Predictor",
    description="""
    Enter client information to predict credit card default probability.
    
    Guidelines:
    - Payment Status: -1=paid early, 0=paid on time, 1-9=payment delay
    - Education: 1=graduate, 2=university, 3=high school, 4=others
    - Marriage: 1=married, 2=single, 3=others
    """
)

# Launch the app
if __name__ == "__main__":
    demo.launch()


# In[ ]:




