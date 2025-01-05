import pandas as pd

def clean_pit_stops_data(df):
    """Clean pit stops data and save to new CSV"""
    # Convert duration to numeric
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    
    # Remove any rows with missing values in key columns
    df = df.dropna(subset=['duration', 'driverId', 'Season', 'Round'])
    
    # Filter durations
    df = df[
        (df['duration'] > 0) & 
        (df['duration'] < 100)  # Assuming no pit stop should take more than 100 seconds
    ]
    
    # Convert numeric columns
    numeric_columns = ['lap', 'stop', 'Season', 'Round']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    # Keep only necessary columns
    columns_to_keep = ['driverId', 'lap', 'stop', 'duration', 'Season', 'Round']
    df = df[columns_to_keep]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Read original data
    pit_stops = pd.read_csv('combined_pit_stops.csv')
    
    # Clean the data
    cleaned_pit_stops = clean_pit_stops_data(pit_stops)
    
    # Save to new CSV
    cleaned_pit_stops.to_csv('cleaned_pit_stops.csv', index=False)
    
    # Print some statistics
    print(f"Total pit stops after cleaning: {len(cleaned_pit_stops)}")
    print(f"Average pit stop duration: {cleaned_pit_stops['duration'].mean():.2f} seconds")
    print(f"Number of unique drivers: {cleaned_pit_stops['driverId'].nunique()}") 