import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("data/diabetic_data.csv")

# Drop duplicates and irrelevant columns
df = df.drop(columns=['encounter_id', 'patient_nbr'])

# Target variable
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Handle missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.5, inplace=True)
df.fillna("unknown", inplace=True)

# Encode categorical features
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Feature-target split
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (choose one)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump({
    'model': model,
    'features': X_train.columns.tolist()
}, "model/readmission_model.pkl")

