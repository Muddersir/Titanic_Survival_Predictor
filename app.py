# app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model = joblib.load('best_titanic_model.pkl')

def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    """
    Predict survival probability for Titanic passenger
    
    Parameters:
    -----------
    Pclass: Passenger class (1, 2, 3)
    Sex: Gender (male, female)
    Age: Age in years
    SibSp: Number of siblings/spouses aboard
    Parch: Number of parents/children aboard
    Fare: Passenger fare
    Embarked: Port of embarkation (C, Q, S)
    
    Returns:
    --------
    str: Prediction result with probability
    """
    try:
        # Create input dataframe with exact feature names
        input_df = pd.DataFrame([[
            Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
        ]],
        columns=[
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
        ])
        
        # Create derived features (same as in training)
        input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch'] + 1
        input_df['IsAlone'] = 0
        input_df.loc[input_df['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Add other engineered features with default values
        input_df['Has_Cabin'] = 0  # Assuming no cabin info
        
        # Title based on gender
        input_df['Title'] = 'Mr' if Sex == 'male' else 'Miss'
        
        # Age groups (same logic as training)
        if Age <= 12:
            input_df['AgeGroup'] = 'Child'
        elif Age <= 18:
            input_df['AgeGroup'] = 'Teen'
        elif Age <= 35:
            input_df['AgeGroup'] = 'Young Adult'
        elif Age <= 60:
            input_df['AgeGroup'] = 'Adult'
        else:
            input_df['AgeGroup'] = 'Senior'
        
        # Fare groups
        if Fare <= 7.91:
            input_df['FareGroup'] = 'Low'
        elif Fare <= 14.45:
            input_df['FareGroup'] = 'Medium'
        elif Fare <= 31.0:
            input_df['FareGroup'] = 'High'
        else:
            input_df['FareGroup'] = 'Very High'
        
        # Ensure categorical columns are properly typed
        categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')
        
        # Get probability prediction
        prob_survival = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        # Format output
        if prediction == 1:
            emoji = "🎉"
            status = "SURVIVED"
            color = "#10B981"  # Green
            confidence_level = "HIGH" if prob_survival > 0.7 else "MEDIUM"
        else:
            emoji = "⚰️"
            status = "DID NOT SURVIVE"
            color = "#EF4444"  # Red
            confidence_level = "HIGH" if prob_survival < 0.3 else "MEDIUM"
        
        # Create detailed output
        output_html = f"""
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 48px; margin-bottom: 10px;'>{emoji}</div>
            <h2 style='color: {color}; margin-bottom: 10px;'>{status}</h2>
            
            <div style='background-color: #F3F4F6; padding: 15px; border-radius: 10px; margin: 15px 0;'>
                <h3 style='margin-top: 0;'>Survival Probability: <span style='color: {color};'>{prob_survival:.1%}</span></h3>
                <div style='background-color: #E5E7EB; height: 20px; border-radius: 10px; margin: 10px 0;'>
                    <div style='background-color: {color}; height: 100%; width: {prob_survival*100}%; border-radius: 10px;'></div>
                </div>
                <p>Confidence: {confidence_level}</p>
            </div>
            
            <div style='background-color: #FEF3C7; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='margin-top: 0;'>Key Factors:</h4>
                <ul style='text-align: left;'>
                    <li>Class {Pclass} {'🏰' if Pclass==1 else '🏠' if Pclass==2 else '🚢'}</li>
                    <li>{'👩' if Sex=='female' else '👨'} {Sex.title()}</li>
                    <li>Age: {Age} years</li>
                    <li>Fare: £{Fare:.2f}</li>
                    <li>Family Size: {input_df['FamilySize'].iloc[0]}</li>
                </ul>
            </div>
        </div>
        """
        
        # Also return plain text for compatibility
        plain_text = f"{emoji} Passenger would have {status.lower()} with {prob_survival:.1%} probability"
        
        return output_html, plain_text, prob_survival
        
    except Exception as e:
        error_msg = f"⚠️ Error making prediction: {str(e)}"
        return error_msg, error_msg, 0.5

# Create the Gradio interface
with gr.Blocks(title="🚢 Titanic Survival Predictor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🚢 Titanic Survival Predictor
    
    Predict whether a passenger would have survived the Titanic disaster based on their characteristics.
    This model uses machine learning trained on historical passenger data.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Passenger Information Section
            gr.Markdown("### Passenger Information")
            
            Pclass = gr.Dropdown(
                choices=[1, 2, 3],
                value=3,
                label="Passenger Class",
                info="1 = First Class, 2 = Second Class, 3 = Third Class"
            )
            
            Sex = gr.Radio(
                choices=["male", "female"],
                value="male",
                label="Gender"
            )
            
            Age = gr.Slider(
                minimum=0,
                maximum=100,
                value=30,
                step=1,
                label="Age (years)"
            )
            
            SibSp = gr.Slider(
                minimum=0,
                maximum=10,
                value=0,
                step=1,
                label="Siblings/Spouses Aboard"
            )
            
            Parch = gr.Slider(
                minimum=0,
                maximum=10,
                value=0,
                step=1,
                label="Parents/Children Aboard"
            )
            
            Fare = gr.Slider(
                minimum=0,
                maximum=200,
                value=32.20,
                step=0.1,
                label="Fare (£)"
            )
            
            Embarked = gr.Radio(
                choices=["C", "Q", "S"],
                value="S",
                label="Port of Embarkation",
                info="C = Cherbourg, Q = Queenstown, S = Southampton"
            )
            
            submit_btn = gr.Button("Predict Survival", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=1):
            # Results Section
            gr.Markdown("### Prediction Results")
            
            # HTML output for rich display
            html_output = gr.HTML(
                label="Prediction Result",
                value="<div style='text-align: center; padding: 50px; color: #6B7280;'>Enter passenger details and click 'Predict Survival'</div>"
            )
            
            # Text output for simple display
            text_output = gr.Textbox(
                label="Prediction Summary",
                interactive=False
            )
            
            # Hidden probability for advanced use
            prob_output = gr.Number(
                label="Survival Probability",
                visible=False
            )
            
            # Probability gauge
            with gr.Accordion("📊 Detailed Statistics", open=False):
                gr.Markdown("""
                **Historical Facts:**
                - Overall survival rate: 38.2%
                - Female survival rate: 74.2%
                - Male survival rate: 18.9%
                - First class survival rate: 62.9%
                """)
    
    # Examples Section
    with gr.Accordion("📋 Example Passengers", open=False):
        gr.Markdown("""
        Try these example passenger profiles:
        """)
        
        examples = gr.Examples(
            examples=[
                [1, "female", 29, 0, 0, 211.34, "C"],  # Upper class female
                [3, "male", 25, 0, 0, 7.25, "S"],      # Lower class male
                [2, "female", 14, 1, 2, 30.5, "C"],    # Family with children
                [1, "male", 45, 1, 0, 50, "S"]         # Upper class male
            ],
            inputs=[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked],
            outputs=[html_output, text_output, prob_output],
            fn=predict_survival,
            cache_examples=True
        )
    
    # Footer
    gr.Markdown("""
    ---
    *Note: This is a machine learning model trained on historical Titanic passenger data. 
    The predictions are based on patterns in the training data and may not reflect individual outcomes.*
    
    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Accuracy: ~82%
    - Features used: 14 engineered features
    """)
    
    # Set up button actions
    submit_btn.click(
        fn=predict_survival,
        inputs=[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked],
        outputs=[html_output, text_output, prob_output]
    )
    
    clear_btn.click(
        fn=lambda: [
            "<div style='text-align: center; padding: 50px; color: #6B7280;'>Enter passenger details and click 'Predict Survival'</div>",
            "",
            0.5
        ],
        inputs=[],
        outputs=[html_output, text_output, prob_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )