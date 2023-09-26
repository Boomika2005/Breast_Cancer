import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer



your_dataset = pd.read_csv("C:\\Users\\BOOMI\\OneDrive\\Desktop\\Breast_cancer.csv")


imputer = SimpleImputer(strategy='mean')
your_dataset = your_dataset.drop(columns=["Unnamed: 32"])  # Drop any extra column if present
your_dataset['diagnosis'] = your_dataset['diagnosis'].map({'M': 1, 'B': 0})  # Map diagnosis to binary values


X = your_dataset.drop(columns=['diagnosis'])  
y = your_dataset['diagnosis'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)


feature_importance = model.feature_importances_
feature_names = your_dataset.columns[:-1]  

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
