import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('data/dataset.csv')

# Select important features
selected_features = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'count', 'srv_count', 'same_srv_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'class'
]
data = data[selected_features]

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Encode categorical features
protocol_type_encoder = LabelEncoder()
X['protocol_type'] = protocol_type_encoder.fit_transform(X['protocol_type'])

service_encoder = LabelEncoder()
X['service'] = service_encoder.fit_transform(X['service'])

flag_encoder = LabelEncoder()
X['flag'] = flag_encoder.fit_transform(X['flag'])

# Save encoders
with open('app1/models/protocol_type_encoder.pkl', 'wb') as f:
    pickle.dump(protocol_type_encoder, f)

with open('app1/models/service_encoder.pkl', 'wb') as f:
    pickle.dump(service_encoder, f)

with open('app1/models/flag_encoder.pkl', 'wb') as f:
    pickle.dump(flag_encoder, f)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open('app1/models/preprocessor.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('app1/models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Training complete. Model and encoders saved successfully.")
