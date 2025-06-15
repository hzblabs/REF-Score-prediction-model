import joblib
import pandas as pd


model = joblib.load('ref_score_predictor_set1.joblib')


sample_data = pd.DataFrame({
    'Main panel': ['D'],
    'FTE of submitted staff': [16.8],
    '% of eligible staff submitted': [55.0]
})


prediction = model.predict(sample_data)
print("Predicted REF Score:", prediction[0])
