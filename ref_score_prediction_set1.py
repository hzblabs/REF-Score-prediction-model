
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


excel_file = pd.ExcelFile("REF 2021 Results - All - 2022-05-06.xlsx")
df = excel_file.parse('Sheet1', skiprows=5)
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header


df = df[df['Profile'] == 'Overall']


columns = [
    'Main panel',
    'FTE of submitted staff',
    '% of eligible staff submitted',
    '4*', '3*', '2*', '1*', 'Unclassified'
]
df = df[columns].copy()
df[['FTE of submitted staff', '% of eligible staff submitted', '4*', '3*', '2*', '1*', 'Unclassified']] =     df[['FTE of submitted staff', '% of eligible staff submitted', '4*', '3*', '2*', '1*', 'Unclassified']].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)


def get_highest_score(row):
    scores = {'4*': row['4*'], '3*': row['3*'], '2*': row['2*'], '1*': row['1*'], 'Unclassified': row['Unclassified']}
    return max(scores, key=scores.get)

df['Target'] = df.apply(get_highest_score, axis=1)


X = df[['Main panel', 'FTE of submitted staff', '% of eligible staff submitted']]
y = df['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


categorical_features = ['Main panel']
numerical_features = ['FTE of submitted staff', '% of eligible staff submitted']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(), categorical_features),
    ('num', StandardScaler(), numerical_features)
])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


joblib.dump(pipeline, 'ref_score_predictor_set1.joblib')


print("Accuracy:", accuracy)
print(report)
