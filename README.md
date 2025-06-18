# REF Score Prediction (Set 1 Features)

There are two models in this repository, one uses REF 2021 data to predict the highest overall quality rating (4*, 3*, etc.) using basic institutional submission features. The other uses a full-text paper to predict a REF Score.

## Files
- `ref_score_prediction_set1.py`: Code to train and evaluate the model.
- `ref_score_predictor_set1.joblib`: Trained model file.
- `REF 2021 Results - All - 2022-05-06.xlsx`: Raw REF dataset (get it from REF2021 site).
- `Model confidence Sore`: A Sheet to understand the confidence score of the model.
- `Steps in training REF AI`: The steps taken in providing the training dataset and fine tuning the model.
- `distilBert`: Fine-tune script.
- `predict_ref_star`: A test script that takes in full text and predict a score with confidence level.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn joblib
   ...
   
2. Run the script
    Basic Model -----> python ref_score_prediction_set1.py
    Fine-tuned Model -----> python predict_ref_star.py

4. Basic Model Inputs 
     Main panel (A, B, C, D)
     FTE of submitted staff
     % of eligible staff submitted

5. Fine tuned Model Inputs
      Pdf folder (code already extracts text from pdf)

