import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

# Load data
df = pd.read_csv("data/diabetic_data.csv")

# Drop duplicates and irrelevant columns
df = df.drop(columns=['encounter_id', 'patient_nbr'])

# Target variable transformation
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Handle missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
df.fillna("unknown", inplace=True)

# Encode categorical features
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Split into features and target
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
]

best_model = None
best_score = 0

best_model_name = ""

# Evaluate each model
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = model.__class__.__name__

print(f"\n✅ Best Model: {best_model_name} with F1 Score: {best_score:.4f}")

# Save the best model

import cloudpickle

with open("model/readmission_model.pkl", "wb") as f:
    cloudpickle.dump({
        'model': best_model,
        'features': X_train.columns.tolist()
    }, f)
