import pandas as pd

def clean_lap_times_data(df):
    """Clean lap times data and save to new CSV"""
    # Convert time to seconds
    def convert_time_to_seconds(time_str):
        try:
            # Remove any trailing/leading whitespace
            time_str = str(time_str).strip()
            # Split minutes and seconds
            if ':' in time_str:
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            return float(time_str)
        except:
            return None
    
    # Convert time to seconds
    df['time'] = df['time'].apply(convert_time_to_seconds)
    
    # Remove any rows with missing values
    df = df.dropna(subset=['time', 'driverId', 'Season', 'Round'])
    
    # Convert numeric columns
    numeric_columns = ['position', 'number', 'Season', 'Round']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    # Keep only necessary columns
    columns_to_keep = ['driverId', 'position', 'time', 'number', 'Season', 'Round']
    df = df[columns_to_keep]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Read original data
    lap_times = pd.read_csv('combined_lap_times.csv')
    
    # Clean the data
    cleaned_lap_times = clean_lap_times_data(lap_times)
    
    # Save to new CSV
    cleaned_lap_times.to_csv('cleaned_lap_times.csv', index=False)
    
    # Print some statistics
    print(f"Total lap times after cleaning: {len(cleaned_lap_times)}")
    print(f"Average lap time: {cleaned_lap_times['time'].mean():.2f} seconds")
    print(f"Number of unique drivers: {cleaned_lap_times['driverId'].nunique()}") 