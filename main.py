import pandas as pd
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data_set = pd.read_csv('AI Research & Development Fellowship - Assessment Datasets - Question 5 - Machine Learning.csv')

# Add target columns
data_set['Tomorrow'] = data_set['Close'].shift(-1)
data_set['Target'] = (data_set['Tomorrow'] > data_set['Close']).astype(int)

# Define features for the model
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train_data = data_set.iloc[:-100]
test_data = data_set.iloc[-100:]

model.fit(train_data[features], train_data['Target'])
predictions = model.predict(test_data[features])
predictions = pd.Series(predictions, index=test_data.index)
print("Predictions:", predictions)

print("\nPrecision Score: ", precision_score(test_data['Target'], predictions))