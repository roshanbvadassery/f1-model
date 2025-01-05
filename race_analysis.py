import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class F1RaceAnalyzer:
    def __init__(self):
        # Load and preprocess the data
        self.lap_times = pd.read_csv('cleaned_lap_times.csv')  # Use cleaned lap times
        self.pit_stops = pd.read_csv('cleaned_pit_stops.csv')  # Use cleaned pit stops
        self.race_results = pd.read_csv('combined_race_results.csv')
        
        # Clean and validate pit stops data
        self.clean_pit_stops_data()
        
    def clean_pit_stops_data(self):
        """Clean and validate pit stops data"""
        # First convert duration to numeric, replacing any errors with NaN
        self.pit_stops['duration'] = pd.to_numeric(self.pit_stops['duration'], errors='coerce')
        
        # Remove any rows with missing values in key columns
        self.pit_stops = self.pit_stops.dropna(subset=['duration', 'driverId', 'Season', 'Round'])
        
        # Now we can safely filter durations
        self.pit_stops = self.pit_stops[
            (self.pit_stops['duration'] > 0) & 
            (self.pit_stops['duration'] < 100)  # Assuming no pit stop should take more than 100 seconds
        ]
        
        # Ensure all numeric columns are properly typed
        numeric_columns = ['lap', 'stop', 'Season', 'Round']
        for col in numeric_columns:
            self.pit_stops[col] = pd.to_numeric(self.pit_stops[col], errors='coerce')
        
        # Remove any remaining rows with NaN values
        self.pit_stops = self.pit_stops.dropna()
        
        # Reset index after cleaning
        self.pit_stops = self.pit_stops.reset_index(drop=True)
        
        # Print some statistics about the cleaned data
        print(f"Total pit stops after cleaning: {len(self.pit_stops)}")
        print(f"Average pit stop duration: {self.pit_stops['duration'].mean():.2f} seconds")
        print(f"Number of unique drivers: {self.pit_stops['driverId'].nunique()}")
    
    def analyze_pit_stop_impact(self):
        """Analyze how pit stop strategy affects final position"""
        # First, count pit stops per driver per race
        pit_counts = self.pit_stops.groupby(['driverId', 'Season', 'Round']).size().reset_index(name='pit_count')
        
        # Then calculate average duration
        pit_durations = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['duration'].mean().reset_index(name='avg_duration')
        
        # Merge pit counts and durations
        pit_stats = pd.merge(
            pit_counts,
            pit_durations,
            on=['driverId', 'Season', 'Round']
        )
        
        # Get relevant columns from race results and rename driverId column
        race_data = self.race_results[['Driver.driverId', 'Season', 'Round', 'position', 'points']]
        race_data = race_data.rename(columns={'Driver.driverId': 'driverId'})
        race_data['position'] = pd.to_numeric(race_data['position'], errors='coerce')
        
        # Merge with race results
        merged_data = pd.merge(
            pit_stats,
            race_data,
            on=['driverId', 'Season', 'Round']
        )
        
        # Calculate correlation only for valid numeric data
        valid_data = merged_data.dropna(subset=['pit_count', 'position'])
        correlation = stats.pearsonr(valid_data['pit_count'], valid_data['position'])
        
        pit_stop_analysis = {
            'pit_stop_count_correlation': correlation[0],
            'p_value': correlation[1],
            'avg_position_by_stops': merged_data.groupby('pit_count')['position'].mean()
        }
        
        return pit_stop_analysis
    
    def analyze_qualifying_impact(self):
        """Analyze correlation between grid position and final position"""
        # Convert grid and position to numeric if they aren't already
        qual_data = self.race_results[['grid', 'position', 'points']].copy()
        qual_data['grid'] = pd.to_numeric(qual_data['grid'], errors='coerce')
        qual_data['position'] = pd.to_numeric(qual_data['position'], errors='coerce')
        
        # Drop any NaN values before correlation
        valid_data = qual_data.dropna()
        grid_correlation = stats.pearsonr(valid_data['grid'], valid_data['position'])
        
        return {
            'grid_position_correlation': grid_correlation[0],
            'p_value': grid_correlation[1],
            'position_improvement': (valid_data['grid'] - valid_data['position']).mean()
        }
    
    def analyze_lap_time_consistency(self):
        """Analyze impact of consistent lap times on race outcome"""
        # Calculate statistics separately to avoid nested column names
        mean_times = self.lap_times.groupby(['driverId', 'Season', 'Round'])['time'].mean().reset_index(name='mean_lap_time')
        std_times = self.lap_times.groupby(['driverId', 'Season', 'Round'])['time'].std().reset_index(name='std_lap_time')
        
        # Merge the statistics
        lap_stats = mean_times.merge(std_times, on=['driverId', 'Season', 'Round'])
        
        # Get relevant columns from race results
        race_data = self.race_results[['Driver.driverId', 'Season', 'Round', 'position']]
        race_data = race_data.rename(columns={'Driver.driverId': 'driverId'})
        race_data['position'] = pd.to_numeric(race_data['position'], errors='coerce')
        
        # Merge with race results
        results = pd.merge(
            lap_stats,
            race_data,
            on=['driverId', 'Season', 'Round']
        )
        
        # Drop NaN values before correlation
        valid_data = results.dropna(subset=['std_lap_time', 'position'])
        consistency_correlation = stats.pearsonr(valid_data['std_lap_time'], valid_data['position'])
        
        return {
            'consistency_correlation': consistency_correlation[0],
            'p_value': consistency_correlation[1],
            'avg_lap_time_correlation': stats.pearsonr(valid_data['mean_lap_time'], valid_data['position'])[0]
        }
    
    def analyze_pit_stop_timing(self):
        """Analyze the impact of pit stop timing on race outcome"""
        # Calculate statistics separately to avoid nested column names
        first_stops = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['lap'].first().reset_index(name='first_stop_lap')
        mean_stops = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['lap'].mean().reset_index(name='mean_stop_lap')
        std_stops = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['lap'].std().reset_index(name='std_stop_lap')
        mean_duration = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['duration'].mean().reset_index(name='mean_duration')
        std_duration = self.pit_stops.groupby(['driverId', 'Season', 'Round'])['duration'].std().reset_index(name='std_duration')
        
        # Merge all statistics
        timing_stats = first_stops.merge(mean_stops, on=['driverId', 'Season', 'Round'])
        timing_stats = timing_stats.merge(std_stops, on=['driverId', 'Season', 'Round'])
        timing_stats = timing_stats.merge(mean_duration, on=['driverId', 'Season', 'Round'])
        timing_stats = timing_stats.merge(std_duration, on=['driverId', 'Season', 'Round'])
        
        # Get race results data
        race_data = self.race_results[['Driver.driverId', 'Season', 'Round', 'position']]
        race_data = race_data.rename(columns={'Driver.driverId': 'driverId'})
        race_data['position'] = pd.to_numeric(race_data['position'], errors='coerce')
        
        # Merge with race results
        timing_analysis = pd.merge(
            timing_stats,
            race_data,
            on=['driverId', 'Season', 'Round']
        )
        
        # Calculate correlations
        valid_data = timing_analysis.dropna()
        
        return {
            'first_stop_correlation': stats.pearsonr(
                valid_data['first_stop_lap'],
                valid_data['position']
            )[0],
            'duration_consistency_impact': stats.pearsonr(
                valid_data['std_duration'],
                valid_data['position']
            )[0],
            'avg_stop_lap_correlation': stats.pearsonr(
                valid_data['mean_stop_lap'],
                valid_data['position']
            )[0]
        }
    
    def generate_visualizations(self):
        """Generate key visualizations for the analysis"""
        plt.figure(figsize=(15, 10))
        
        # Pit stop analysis plot
        pit_data = self.analyze_pit_stop_impact()
        plt.subplot(2, 2, 1)
        
        # Convert to DataFrame for easier plotting
        pit_positions = pit_data['avg_position_by_stops'].reset_index()
        pit_positions.columns = ['Number of Stops', 'Average Position']
        
        sns.barplot(data=pit_positions, x='Number of Stops', y='Average Position')
        plt.title('Average Position by Number of Pit Stops')
        
        # Add qualifying impact plot
        plt.subplot(2, 2, 2)
        qual_data = self.race_results[['grid', 'position']].copy()
        qual_data = qual_data.apply(pd.to_numeric, errors='coerce').dropna()
        sns.scatterplot(data=qual_data, x='grid', y='position', alpha=0.5)
        plt.title('Grid Position vs Final Position')
        
        # Save plots
        plt.tight_layout()
        plt.savefig('race_analysis_plots.png')
        
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        try:
            pit_analysis = self.analyze_pit_stop_impact()
            timing_analysis = self.analyze_pit_stop_timing()
            qual_analysis = self.analyze_qualifying_impact()
            consistency_analysis = self.analyze_lap_time_consistency()
            
            report = {
                'pit_stop_analysis': pit_analysis,
                'timing_analysis': timing_analysis,
                'qualifying_analysis': qual_analysis,
                'consistency_analysis': consistency_analysis,
                'key_findings': []
            }
            
            # Add key findings with more detailed insights
            if abs(pit_analysis['pit_stop_count_correlation']) > 0.3:
                report['key_findings'].append(
                    f"Strong correlation ({pit_analysis['pit_stop_count_correlation']:.2f}) "
                    "between number of pit stops and final position"
                )
            
            if abs(timing_analysis['first_stop_correlation']) > 0.3:
                report['key_findings'].append(
                    f"First pit stop timing shows {timing_analysis['first_stop_correlation']:.2f} "
                    "correlation with final position"
                )
            
            if abs(consistency_analysis['consistency_correlation']) > 0.3:
                report['key_findings'].append(
                    f"Lap time consistency shows {consistency_analysis['consistency_correlation']:.2f} "
                    "correlation with final position"
                )
            
            return report
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = F1RaceAnalyzer()
    report = analyzer.generate_report()
    
    if report:
        analyzer.generate_visualizations()
        
        print("\nKey Race Performance Factors Analysis:")
        print("======================================")
        for finding in report['key_findings']:
            print(f"- {finding}")
    else:
        print("Failed to generate report due to errors in data processing.") 