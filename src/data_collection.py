"""
Data Collection for ABS Challenge Strategy
Pulls MLB Statcast pitch-by-pitch data via pybaseball

Note: We use MLB data to simulate the ABS challenge environment.
While the challenge system was piloted in Triple-A, MLB Statcast 
provides the richest pitch tracking data available. Our framework
is generalizable to any league implementing ABS.
"""

import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from pybaseball import statcast, cache
    cache.enable()
    print("pybaseball imported successfully")
except ImportError:
    print("ERROR: Install pybaseball first:")
    print("  pip install pybaseball --break-system-packages")
    exit(1)


def collect_mlb_data(start_date, end_date, output_file):
    """
    Collect MLB Statcast pitch data.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str  
        End date in 'YYYY-MM-DD' format
    output_file : str
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame or None
    """
    print("COLLECTING MLB PITCH DATA")
    print(f"Date range: {start_date} to {end_date}")
    print("\nThis will take 10-30 minutes for a full season...")
    print("Progress bar will show as data loads.\n")
    
    try:
        # Get the data from Baseball Savant via pybaseball
        data = statcast(start_dt=start_date, end_dt=end_date)
        
        if data is None or len(data) == 0:
            print("No data returned!")
            return None
        
        print(f"\nRetrieved {len(data):,} pitches")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        data.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        # Quick summary
        
        print("DATA SUMMARY")
        
        print(f"Total pitches: {len(data):,}")
        print(f"Date range: {data['game_date'].min()} to {data['game_date'].max()}")
        print(f"Unique games: {data['game_pk'].nunique():,}")
        print(f"File size: ~{os.path.getsize(output_file) / 1_000_000:.1f} MB")
        
        # Pitch type breakdown
        if 'description' in data.columns:
            print(f"\nPitch outcomes (top 10):")
            desc_counts = data['description'].value_counts().head(10)
            for desc, count in desc_counts.items():
                print(f"  {desc}: {count:,} ({count/len(data)*100:.1f}%)")
        
        return data
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution"""
    
    
    print("MLB PITCH DATA COLLECTION")
    print("For ABS Challenge Strategy Research")
    
    
    # CONFIGURATION - Choose date range
    
    # OPTION 1: Test dataset (1 week) - Use first to verify pipeline works
    test_config = {
        'start': '2024-07-01',
        'end': '2024-07-07',
        'output': '../data/raw/mlb_2024_test.csv'
    }
    
    # OPTION 2: Full dataset (4 months of 2024 season)
    full_config = {
        'start': '2024-06-01',
        'end': '2024-09-30',
        'output': '../data/raw/mlb_2024_full.csv'
    }
    
    # SELECT WHICH CONFIG TO USE
    
    # Uncomment ONE of these:
    # config = test_config    # Start here to test
    config = full_config      # Use for full analysis
    
    
    print(f"\nConfiguration:")
    print(f"  Start date: {config['start']}")
    print(f"  End date: {config['end']}")
    print(f"  Output: {config['output']}")
    
    # Collect the data
    data = collect_mlb_data(
        start_date=config['start'],
        end_date=config['end'],
        output_file=config['output']
    )
    
    if data is not None:
        print("DATA COLLECTION COMPLETE!")
        print("\nNext step: Run data_preprocessing.py")
        print("  cd src")
        print("  python data_preprocessing.py")
    else:
        print("\nCollection failed - check error above")


if __name__ == "__main__":
    main()