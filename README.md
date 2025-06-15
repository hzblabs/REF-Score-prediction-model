# REF Score Prediction (Set 1 Features)

This project uses REF 2021 data to predict the highest overall quality rating (4*, 3*, etc.) using basic institutional submission features.

## Files
- `ref_score_prediction_set1.py`: Code to train and evaluate the model
- `ref_score_predictor_set1.joblib`: Trained model file
- `REF 2021 Results - All - 2022-05-06.xlsx`: Raw REF dataset

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn joblib

2. Run the script
     python ref_score_prediction_set1.py

3. Model Inputs 
     Main panel (A, B, C, D)
     FTE of submitted staff
     % of eligible staff submitted

