import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

class F1Predictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
    def prepare_data(self):
        # Load cleaned data
        lap_times = pd.read_csv('cleaned_lap_times.csv')
        pit_stops = pd.read_csv('cleaned_pit_stops.csv')
        race_results = pd.read_csv('combined_race_results.csv')
        
        # Calculate average lap times and consistency per race
        lap_stats = lap_times.groupby(['driverId', 'Season', 'Round']).agg({
            'time': ['mean', 'std']
        }).reset_index()
        lap_stats.columns = ['driverId', 'Season', 'Round', 'avg_lap_time', 'lap_time_consistency']
        
        # Calculate pit stop statistics
        pit_stats = pit_stops.groupby(['driverId', 'Season', 'Round']).agg({
            'duration': ['count', 'mean', 'std']
        }).reset_index()
        pit_stats.columns = ['driverId', 'Season', 'Round', 'pit_stops', 'avg_pit_time', 'pit_consistency']
        
        # Get relevant columns from race results
        race_data = race_results[['Driver.driverId', 'Season', 'Round', 'grid', 'position', 'points']]
        race_data = race_data.rename(columns={'Driver.driverId': 'driverId'})
        
        # Merge all data
        merged_data = race_data.merge(lap_stats, on=['driverId', 'Season', 'Round'], how='left')
        merged_data = merged_data.merge(pit_stats, on=['driverId', 'Season', 'Round'], how='left')
        
        # Fill missing values
        merged_data = merged_data.fillna({
            'pit_stops': 0,
            'avg_pit_time': merged_data['avg_pit_time'].mean(),
            'pit_consistency': merged_data['pit_consistency'].mean(),
            'lap_time_consistency': merged_data['lap_time_consistency'].mean()
        })
        
        # Encode categorical variables
        for col in ['driverId']:
            le = LabelEncoder()
            merged_data[col] = le.fit_transform(merged_data[col])
            self.label_encoders[col] = le
        
        # Define features and target
        features = ['driverId', 'grid', 'avg_lap_time', 'lap_time_consistency', 
                   'pit_stops', 'avg_pit_time', 'pit_consistency']
        
        X = merged_data[features]
        y = merged_data['points']  # or 'position' if you prefer to predict position
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, model_type='random_forest'):
        """Train the model using Random Forest by default"""
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use Random Forest as the default model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training Score: {train_score:.4f}")
        print(f"Testing Score: {test_score:.4f}")
        
        # Print feature importances
        feature_names = ['Driver', 'Grid Position', 'Avg Lap Time', 
                        'Lap Consistency', 'Pit Stops', 'Avg Pit Time', 
                        'Pit Consistency']
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        print("\nFeature Importances:")
        print(importances.sort_values('importance', ascending=False))
        
        return test_score
    
    def save_model(self, filename='f1_predictor.joblib'):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='f1_predictor.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        print("Model loaded successfully")
    
    def predict(self, driver_id, grid_position, avg_lap_time, lap_consistency,
               pit_stops, avg_pit_time, pit_consistency):
        """Make a prediction for a single race"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Encode driver ID
        encoded_driver = self.label_encoders['driverId'].transform([driver_id])[0]
        
        # Create feature array
        features = np.array([[
            encoded_driver, grid_position, avg_lap_time, lap_consistency,
            pit_stops, avg_pit_time, pit_consistency
        ]])
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        
        return prediction

if __name__ == "__main__":
    # Create and train the model
    predictor = F1Predictor()
    score = predictor.train_model()
    
    # Save the model
    predictor.save_model()
    
    # Example prediction
    prediction = predictor.predict(
        driver_id='max_verstappen',
        grid_position=1,
        avg_lap_time=80.0,  # in seconds
        lap_consistency=0.5,
        pit_stops=2,
        avg_pit_time=24.0,
        pit_consistency=0.3
    )
    
    print(f"\nPredicted points: {prediction:.2f}") 