import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class F1StrategyAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        # Load the CSV files
        self.lap_times = pd.read_csv('combined_lap_times.csv')
        self.pit_stops = pd.read_csv('combined_pit_stops.csv')
        self.race_results = pd.read_csv('combined_race_results.csv')
        
        # Merge the dataframes
        self.data = pd.merge(self.lap_times, self.pit_stops, 
                           on=['driverId', 'Season', 'Round'], how='left')
        self.data = pd.merge(self.data, self.race_results,
                           on=['driverId', 'Season', 'Round'], how='left')
        
        # Convert lap times to seconds
        self.data['time'] = pd.to_timedelta(self.data['time']).dt.total_seconds()
        
        # Encode categorical variables
        categorical_columns = ['driverId', 'status', 'Constructor.constructorId']
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature selection
        self.features = ['position', 'number', 'grid', 'time', 
                        'driverId', 'Constructor.constructorId']
        self.target = 'points'
        
        # Split the data
        self.X = self.data[self.features]
        self.y = self.data[self.target]
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        return self.X_scaled, self.y
    
    def random_forest_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        rf_score = r2_score(y_test, y_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return rf_model, rf_score, rf_rmse
    
    def xgboost_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred = xgb_model.predict(X_test)
        xgb_score = r2_score(y_test, y_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return xgb_model, xgb_score, xgb_rmse
    
    def lstm_model(self):
        # Reshape data for LSTM (samples, time steps, features)
        X_reshaped = self.X_scaled.reshape((self.X_scaled.shape[0], 1, self.X_scaled.shape[1]))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, self.y, test_size=0.2, random_state=42
        )
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, self.X_scaled.shape[1])),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        y_pred = model.predict(X_test)
        lstm_score = r2_score(y_test, y_pred)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, lstm_score, lstm_rmse

# Usage example
if __name__ == "__main__":
    analyzer = F1StrategyAnalyzer()
    X, y = analyzer.load_and_preprocess_data()
    
    # Train and evaluate all models
    rf_model, rf_score, rf_rmse = analyzer.random_forest_model()
    xgb_model, xgb_score, xgb_rmse = analyzer.xgboost_model()
    lstm_model, lstm_score, lstm_rmse = analyzer.lstm_model()
    
    # Print results
    print("\nModel Performance Comparison:")
    print(f"Random Forest - R2 Score: {rf_score:.4f}, RMSE: {rf_rmse:.4f}")
    print(f"XGBoost - R2 Score: {xgb_score:.4f}, RMSE: {xgb_rmse:.4f}")
    print(f"LSTM - R2 Score: {lstm_score:.4f}, RMSE: {lstm_rmse:.4f}") 