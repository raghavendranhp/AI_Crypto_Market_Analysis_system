import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import nbformat as nbf

#this is the notebook generation code
nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell('# AI Crypto Market Signal & Risk Analyzer - Phase 1: EDA & Modeling\n\nin this notebook, we perform eda, clean the data, engineer features, and train the random forest and isolation forest models.'))

code_cell_1 = """#step 1: load data and perform eda
import pandas as pd
import numpy as np

#load the data
df = pd.read_csv('../data/crypto_trading_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_1))

code_cell_2 = """#step 2: feature engineering
#calculate price change %
df['price_change_pct'] = df['close_price'].pct_change() * 100

#calculate ma7 and ma30
df['ma7'] = df['close_price'].rolling(window=7).mean()
df['ma30'] = df['close_price'].rolling(window=30).mean()

#calculate volatility index (standard deviation of price change)
df['volatility_index'] = df['price_change_pct'].rolling(window=7).std()

#calculate volume spike indicator (ratio of volume to 7-period average volume)
df['vol_ma7'] = df['volume'].rolling(window=7).mean()
df['volume_spike_ratio'] = df['volume'] / df['vol_ma7']

#drop nan values due to rolling windows
df.dropna(inplace=True)
print(df.head())
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_2))

code_cell_3 = """#step 3: train random forest for trend classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#define target: 1 for bullish (price goes up next period), -1 for bearish, 0 for neutral
df['next_period_change'] = df['close_price'].shift(-1) - df['close_price']
def categorize_trend(change):
    if change > 0:
        return 'Bullish'
    elif change < 0:
        return 'Bearish'
    else:
        return 'Neutral'

df['target_trend'] = df['next_period_change'].apply(categorize_trend)
df.dropna(inplace=True)

#select features
features = ['price_change_pct', 'ma7', 'volatility_index', 'volume_spike_ratio']
x = df[features]
y = df['target_trend']

#train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

#save the model
joblib.dump(rf_model, '../app/models/random_forest_trend.joblib')
print("random forest model saved successfully.")
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_3))

code_cell_4 = """#step 4: math-based volatility risk score (0-100)
#formula: min(100, (volatility_index * 10) + (volume_spike_ratio * 5))
df['risk_score'] = np.minimum(100, (df['volatility_index'] * 10) + (df['volume_spike_ratio'] * 5))
print(df[['timestamp', 'risk_score']].head())
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_4))

code_cell_5 = """#step 5: train isolation forest for anomaly detection
from sklearn.ensemble import IsolationForest

#we will use the selected features plus risk score for anomaly detection
anomaly_features = ['price_change_pct', 'ma7', 'volatility_index', 'volume_spike_ratio', 'risk_score']
x_anomaly = df[anomaly_features]

#train the model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(x_anomaly)

#predict anomalies (-1 for anomaly, 1 for normal)
df['anomaly'] = iso_forest.predict(x_anomaly)

#step 6: save the isolation forest model
joblib.dump(iso_forest, '../app/models/isolation_forest_anomaly.joblib')
print("isolation forest model saved successfully.")
"""
nb.cells.append(nbf.v4.new_code_cell(code_cell_5))

#write the notebook
with open('notebooks/01_eda_and_modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("notebook generated at notebooks/01_eda_and_modeling.ipynb")

#execute the code to actually generate the models
df = pd.read_csv('data/crypto_trading_dataset.csv')
df['price_change_pct'] = df['close_price'].pct_change() * 100
df['ma7'] = df['close_price'].rolling(window=7).mean()
df['ma30'] = df['close_price'].rolling(window=30).mean()
df['volatility_index'] = df['price_change_pct'].rolling(window=7).std()
df['vol_ma7'] = df['volume'].rolling(window=7).mean()
df['volume_spike_ratio'] = df['volume'] / df['vol_ma7']
df.dropna(inplace=True)

df['next_period_change'] = df['close_price'].shift(-1) - df['close_price']
def categorize_trend(change):
    if change > 0: return 'Bullish'
    elif change < 0: return 'Bearish'
    else: return 'Neutral'

df['target_trend'] = df['next_period_change'].apply(categorize_trend)
df.dropna(inplace=True)

features = ['price_change_pct', 'ma7', 'volatility_index', 'volume_spike_ratio']
x = df[features]
y = df['target_trend']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x, y)
joblib.dump(rf_model, 'app/models/random_forest_trend.joblib')

df['risk_score'] = np.minimum(100, (df['volatility_index'] * 10) + (df['volume_spike_ratio'] * 5))

anomaly_features = ['price_change_pct', 'ma7', 'volatility_index', 'volume_spike_ratio', 'risk_score']
x_anomaly = df[anomaly_features]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(x_anomaly)
joblib.dump(iso_forest, 'app/models/isolation_forest_anomaly.joblib')

print("models successfully trained and saved!")
