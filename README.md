# REF Score Prediction (Set 1 Features)

There are two models in this repository, one uses REF 2021 data to predict the highest overall quality rating (4*, 3*, etc.) using basic institutional submission features. The other uses a full-text paper to predict a REF Score.

## Files
- `ref_score_prediction_set1.py`: Code to train and evaluate the model.
- `ref_score_predictor_set1.joblib`: Trained model file.
- `Model confidence Sore`: A Sheet to understand the confidence score of the model.
- `Steps in training REF AI`: The steps taken in providing the training dataset and fine tuning the model.
- `manual.pdf`: A sheet to understand the chosen procedure for labelling and training.
- `distilBert`: Fine-tune script.
- `Test.py`: A test script that takes in set 1 procedures features to predict a score (Version 1 Model).
- `predict_ref_star`: A test script that takes in full text and predict a score with confidence level(Version 2 Model).

## How to Run (Version 1&2 Model)
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn joblib
   
2. Run the script
    Basic Model -----> python ref_score_prediction_set1.py
    Fine-tuned Model -----> python predict_ref_star.py

3. Basic Model Inputs (Version 1) 
     Main panel (A, B, C, D)
     FTE of submitted staff
     % of eligible staff submitted

4. Fine tuned Model Inputs (Version 2)
      Pdf folder (code already extracts text from pdf)

