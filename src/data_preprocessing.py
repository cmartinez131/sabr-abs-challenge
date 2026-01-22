"""
Data Preprocessing for ABS Challenge Strategy
This script:
1. Loads raw MLB pitch data
2. Identifies challengeable pitches (taken, close to zone)
3. Calculates proper RE24-based reward values
4. Engineers features for Q-learning state space
5. Creates train/val/test splits

RE24 Implementation based on FanGraphs methodology:
https://www.fangraphs.com/library/misc/re24/

RE24 = (Run Expectancy of ending state) - (Run Expectancy of starting state) + Runs scored
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


# RE24 MATRIX - Run Expectancy by Base-Out State
# Source: FanGraphs / Tango's Run Expectancy Matrix
# These are the expected runs from each base-out state to end of inning
# Run environment: ~4.15-4.30 runs per game (modern MLB average)

RE24_MATRIX = {
    # Format: (runner_on_1b, runner_on_2b, runner_on_3b, outs) -> expected_runs
    
    # 0 outs
    (0, 0, 0, 0): 0.461,  # Bases empty
    (1, 0, 0, 0): 0.831,  # Runner on 1st
    (0, 1, 0, 0): 1.068,  # Runner on 2nd
    (0, 0, 1, 0): 1.426,  # Runner on 3rd
    (1, 1, 0, 0): 1.373,  # 1st & 2nd
    (1, 0, 1, 0): 1.798,  # 1st & 3rd
    (0, 1, 1, 0): 1.920,  # 2nd & 3rd
    (1, 1, 1, 0): 2.282,  # Bases loaded
    
    # 1 out
    (0, 0, 0, 1): 0.243,  # Bases empty
    (1, 0, 0, 1): 0.489,  # Runner on 1st
    (0, 1, 0, 1): 0.644,  # Runner on 2nd
    (0, 0, 1, 1): 0.865,  # Runner on 3rd
    (1, 1, 0, 1): 0.908,  # 1st & 2nd
    (1, 0, 1, 1): 1.140,  # 1st & 3rd
    (0, 1, 1, 1): 1.352,  # 2nd & 3rd
    (1, 1, 1, 1): 1.520,  # Bases loaded
    
    # 2 outs
    (0, 0, 0, 2): 0.095,  # Bases empty
    (1, 0, 0, 2): 0.214,  # Runner on 1st
    (0, 1, 0, 2): 0.305,  # Runner on 2nd
    (0, 0, 1, 2): 0.413,  # Runner on 3rd (FanGraphs: 0.413)
    (1, 1, 0, 2): 0.343,  # 1st & 2nd
    (1, 0, 1, 2): 0.471,  # 1st & 3rd
    (0, 1, 1, 2): 0.570,  # 2nd & 3rd
    (1, 1, 1, 2): 0.736,  # Bases loaded
}


# COUNT-BASED RUN VALUES

# These represent the expected run value change based on count
# Source: Tango's "The Book" / Various sabermetric research
# Positive = favorable for batter, Negative = favorable for pitcher

COUNT_RUN_VALUES = {
    # (balls, strikes): linear weight relative to start of PA
    (0, 0): 0.000,   # Start of at-bat (baseline)
    (1, 0): 0.038,   # 1-0 count favors batter
    (0, 1): -0.043,  # 0-1 count favors pitcher
    (2, 0): 0.076,   # 2-0 batter's count
    (1, 1): -0.004,  # 1-1 roughly even
    (0, 2): -0.096,  # 0-2 pitcher's count
    (3, 0): 0.118,   # 3-0 batter way ahead
    (2, 1): 0.034,   # 2-1 batter ahead
    (1, 2): -0.051,  # 1-2 pitcher ahead
    (3, 1): 0.077,   # 3-1 batter ahead
    (2, 2): -0.015,  # 2-2 roughly even
    (3, 2): 0.040,   # Full count (slight batter advantage)
}


class DataPreprocessor:
    """Preprocesses MLB pitch data for ABS challenge strategy"""
    
    def __init__(self, raw_data_path):
        """Load raw data"""
        print("DATA PREPROCESSING")
        
        print(f"\nLoading data from {raw_data_path}...")
        
        self.df = pd.read_csv(raw_data_path)
        print(f"Loaded {len(self.df):,} pitches")
        
    def identify_challengeable_pitches(self):
        """
        Identify which pitches are challengeable.
        
        Challengeable = Taken pitch (called ball or strike) near zone edge.
        In ABS, teams can only challenge taken pitches, not swings.
        """
        print("STEP 1: Identifying Challengeable Pitches")
        
        # Filter to TAKEN pitches only (called balls and strikes, not swings)
        taken_descriptions = ['called_strike', 'ball', 'blocked_ball']
        taken_mask = self.df['description'].isin(taken_descriptions)
        
        print(f"Total pitches: {len(self.df):,}")
        print(f"Taken pitches (called ball/strike): {taken_mask.sum():,} ({taken_mask.mean()*100:.1f}%)")
        
        # Strike zone boundaries
        # Width: 17 inches = 8.5 inches each side = ±0.708 feet
        ZONE_WIDTH = 17 / 12 / 2  # 0.708 feet from center
        self.df['zone_left'] = -ZONE_WIDTH
        self.df['zone_right'] = ZONE_WIDTH
        # Height: sz_top and sz_bot are batter-specific from Statcast
        
        # Calculate distance from zone edge
        # Horizontal distance (0 if inside zone horizontally)
        h_dist = np.where(
            self.df['plate_x'] < self.df['zone_left'],
            self.df['zone_left'] - self.df['plate_x'],
            np.where(
                self.df['plate_x'] > self.df['zone_right'],
                self.df['plate_x'] - self.df['zone_right'],
                0  # Inside zone horizontally
            )
        )
        
        # Vertical distance (0 if inside zone vertically)
        v_dist = np.where(
            self.df['plate_z'] < self.df['sz_bot'],
            self.df['sz_bot'] - self.df['plate_z'],
            np.where(
                self.df['plate_z'] > self.df['sz_top'],
                self.df['plate_z'] - self.df['sz_top'],
                0  # Inside zone vertically
            )
        )
        
        # Total distance from zone edge (Euclidean)
        self.df['dist_from_zone'] = np.sqrt(h_dist**2 + v_dist**2)
        
        # Challengeable = Taken pitch within 3 inches of zone edge
        # This is a reasonable "close call" threshold that teams would consider
        CHALLENGE_THRESHOLD = 3 / 12  # 3 inches = 0.25 feet
        
        near_zone = self.df['dist_from_zone'] <= CHALLENGE_THRESHOLD
        challengeable = taken_mask & near_zone
        
        # Also need valid zone data
        valid_zone = (
            self.df['plate_x'].notna() & 
            self.df['plate_z'].notna() &
            self.df['sz_top'].notna() & 
            self.df['sz_bot'].notna()
        )
        challengeable = challengeable & valid_zone
        
        print(f"Near zone edge (within 3\"): {(near_zone & taken_mask).sum():,}")
        print(f"Challengeable pitches: {challengeable.sum():,} ({challengeable.mean()*100:.1f}%)")
        
        # Filter to challengeable only
        self.df = self.df[challengeable].copy()
        print(f"\nFiltered to {len(self.df):,} challengeable pitches")
        
    def determine_true_call(self):
        """
        Determine what the call SHOULD have been based on pitch location.
        This lets us identify incorrect calls that could be overturned.
        """
        print("STEP 2: Determining True Call (Ball vs Strike)")
        
        # Is the pitch actually in the strike zone?
        in_zone_h = (self.df['plate_x'] >= self.df['zone_left']) & \
                    (self.df['plate_x'] <= self.df['zone_right'])
        in_zone_v = (self.df['plate_z'] >= self.df['sz_bot']) & \
                    (self.df['plate_z'] <= self.df['sz_top'])
        in_zone = in_zone_h & in_zone_v
        
        # True call based on actual location
        self.df['true_call'] = np.where(in_zone, 'strike', 'ball')
        
        # Actual call from umpire (what was called in the game)
        self.df['actual_call'] = np.where(
            self.df['description'].str.contains('strike', case=False, na=False),
            'strike',
            'ball'
        )
        
        # Is the call incorrect? (opportunity to challenge)
        self.df['call_incorrect'] = self.df['true_call'] != self.df['actual_call']
        
        # Breakdown by type
        called_strike_was_ball = (self.df['actual_call'] == 'strike') & (self.df['true_call'] == 'ball')
        called_ball_was_strike = (self.df['actual_call'] == 'ball') & (self.df['true_call'] == 'strike')
        
        print(f"Incorrect calls: {self.df['call_incorrect'].sum():,} ({self.df['call_incorrect'].mean()*100:.1f}%)")
        print(f"  Called strike, was ball: {called_strike_was_ball.sum():,} (benefits offense if overturned)")
        print(f"  Called ball, was strike: {called_ball_was_strike.sum():,} (benefits defense if overturned)")
        
        # Sanity check: expect ~15-35% incorrect on borderline pitches
        incorrect_rate = self.df['call_incorrect'].mean()
        if incorrect_rate > 0.45:
            print(f"\nWARNING: Incorrect rate {incorrect_rate*100:.1f}% is high.")
            print("    This may indicate zone boundary issues. Expected ~15-35%.")
        elif incorrect_rate < 0.10:
            print(f"\nWARNING: Incorrect rate {incorrect_rate*100:.1f}% is low.")
            print("    Challenge threshold may be too tight.")
        
    def calculate_overturn_probability(self):
        """
        Calculate P(challenge succeeds) based on pitch location.
        
        Farther from zone edge = more obvious miss = higher overturn probability.
        """
        print("STEP 3: Calculating Overturn Probability")
        
        dist_inches = self.df['dist_from_zone'] * 12  # Convert feet to inches
        
        # P(overturn) model:
        # - Call incorrect + very close to edge (<1"): 50% (could go either way)
        # - Call incorrect + medium distance (1-2"): 65% (likely incorrect)  
        # - Call incorrect + far from edge (2-3"): 80% (clear miss)
        # - Call correct: 5% (small chance due to measurement error)
        
        p_overturn_if_incorrect = np.where(
            dist_inches < 1.0, 0.50,
            np.where(dist_inches < 2.0, 0.65, 0.80)
        )
        
        self.df['p_overturn'] = np.where(
            self.df['call_incorrect'],
            p_overturn_if_incorrect,
            0.05  # Small chance even if call appears correct
        )
        
        print(f"Average P(overturn): {self.df['p_overturn'].mean():.3f}")
        print(f"\nP(overturn) for INCORRECT calls by distance:")
        for thresh in [1, 2, 3]:
            mask = (dist_inches < thresh) & self.df['call_incorrect']
            if mask.sum() > 0:
                print(f"  < {thresh}\": {self.df.loc[mask, 'p_overturn'].mean():.3f} (n={mask.sum():,})")
    
    def _get_re24(self, on_1b, on_2b, on_3b, outs):
        """Look up RE24 value for a base-out state"""
        # Convert to binary indicators
        key = (
            1 if on_1b else 0,
            1 if on_2b else 0, 
            1 if on_3b else 0,
            min(int(outs), 2)  # Cap at 2 outs
        )
        return RE24_MATRIX.get(key, 0.0)
    
    def _calculate_walk_re24_change(self, row):
        """
        Calculate RE24 change from a walk.
        Walk: batter reaches 1st, runners advance if forced.
        """
        outs = int(row['outs_when_up'])
        on_1b = bool(row['on_1b'])
        on_2b = bool(row['on_2b'])
        on_3b = bool(row['on_3b'])
        
        # Current state RE24
        current_re24 = self._get_re24(on_1b, on_2b, on_3b, outs)
        
        # After walk: batter on 1st, runners advance if forced
        runs_scored = 0
        new_3b = on_3b
        new_2b = on_2b
        new_1b = True  # Batter takes first
        
        if on_1b:  # Force to 2nd
            new_2b = True
            if on_2b:  # Force to 3rd
                new_3b = True
                if on_3b:  # Force home = run scores
                    runs_scored = 1
        
        # New state RE24
        new_re24 = self._get_re24(new_1b, new_2b, new_3b, outs)
        
        # RE24 formula: new_re24 - current_re24 + runs_scored
        return new_re24 - current_re24 + runs_scored
    
    def _calculate_strikeout_re24_change(self, row):
        """
        Calculate RE24 change from a strikeout.
        Strikeout: one more out recorded, runners stay.
        """
        outs = int(row['outs_when_up'])
        on_1b = bool(row['on_1b'])
        on_2b = bool(row['on_2b'])
        on_3b = bool(row['on_3b'])
        
        # Current state RE24
        current_re24 = self._get_re24(on_1b, on_2b, on_3b, outs)
        
        # After strikeout: one more out (or inning over if 2 outs)
        if outs >= 2:
            new_re24 = 0.0  # Inning over
        else:
            new_re24 = self._get_re24(on_1b, on_2b, on_3b, outs + 1)
        
        # RE24 formula: new_re24 - current_re24 (negative = bad for offense)
        return new_re24 - current_re24
    
    def add_re24_features(self):
        """
        Calculate RE24-based delta for successful challenges.
        
        This is the REWARD FUNCTION for Q-learning!
        
        Key insight: Challenge value depends heavily on count outcome:
        - Most valuable: Strikeout reversals (strike 3 called, was ball)
        - Most valuable: Walk reversals (ball 4 called, was strike)
        - Less valuable: Regular count changes within at-bat
        """
        print("STEP 4: Calculating RE24 Features (Q-Learning Reward)")
        
        # Fill NaN for baserunner columns
        for col in ['on_1b', 'on_2b', 'on_3b']:
            self.df[col] = self.df[col].fillna(0).astype(int)
        
        # Calculate current RE24 state
        self.df['current_re24'] = self.df.apply(
            lambda row: self._get_re24(
                row['on_1b'], row['on_2b'], row['on_3b'], row['outs_when_up']
            ),
            axis=1
        )
        
        def calculate_challenge_value(row):
            """
            Calculate the run value gained if challenge succeeds.
            
            Cases:
            1. Called strike, was ball (offensive challenge):
               - If strike 3: HUGE value (avoid strikeout, AB continues)
               - Otherwise: moderate value (better count)
               
            2. Called ball, was strike (defensive challenge):
               - If ball 4: HUGE value (avoid walk, AB continues)
               - Otherwise: moderate value (better count for pitcher)
            """
            balls = int(row['balls'])
            strikes = int(row['strikes'])
            actual = row['actual_call']
            true = row['true_call']
            
            # Call was correct - no value in challenging
            if actual == true:
                return 0.0
            
            # CASE 1: Called STRIKE, should be BALL (offense benefits)
            
            if actual == 'strike' and true == 'ball':
                
                # The count AFTER the called strike is (balls, strikes+1)
                # But wait - the row shows the count BEFORE this pitch
                # So if strikes=2, this pitch made it strike 3 = strikeout
                
                if strikes == 2:
                    # STRIKEOUT REVERSAL - This is the big one!
                    # Batter was rung up on strike 3, but it should be a ball
                    # AB continues at (balls+1, 2) instead of out recorded
                    
                    strikeout_cost = self._calculate_strikeout_re24_change(row)
                    # strikeout_cost is negative (bad for offense)
                    # Avoiding it means GAINING that value back
                    
                    # New count would be (balls+1, 2)
                    new_balls = balls + 1
                    if new_balls >= 4:
                        # Actually becomes a walk! (K reversed to BB)
                        walk_value = self._calculate_walk_re24_change(row)
                        # Total: avoid strikeout + get walk
                        return -strikeout_cost + walk_value
                    else:
                        # AB continues at (new_balls, 2)
                        # Value = avoiding strikeout + count improvement
                        count_change = (COUNT_RUN_VALUES.get((new_balls, 2), 0) - 
                                       COUNT_RUN_VALUES.get((balls, 2), 0))
                        return -strikeout_cost + count_change
                else:
                    # Regular count change (not strikeout situation)
                    # Count goes from (balls, strikes+1) to (balls+1, strikes)
                    old_count = (balls, min(strikes + 1, 2))
                    new_count = (min(balls + 1, 3), strikes)
                    
                    old_value = COUNT_RUN_VALUES.get(old_count, 0)
                    new_value = COUNT_RUN_VALUES.get(new_count, 0)
                    
                    return new_value - old_value
            
            
            # CASE 2: Called BALL, should be STRIKE (defense benefits)
            
            elif actual == 'ball' and true == 'strike':
                
                # The count AFTER the called ball is (balls+1, strikes)
                # If balls=3, this pitch made it ball 4 = walk
                
                if balls == 3:
                    # WALK REVERSAL - Big for defense!
                    # Walk was given, but it should be a strike
                    # AB continues at (3, strikes+1) instead of walk
                    
                    walk_value = self._calculate_walk_re24_change(row)
                    # walk_value is positive (good for offense)
                    # Reversing it means TAKING that value from offense
                    
                    # New count would be (3, strikes+1)
                    new_strikes = strikes + 1
                    if new_strikes >= 3:
                        # Actually becomes strikeout! (BB reversed to K)
                        strikeout_cost = self._calculate_strikeout_re24_change(row)
                        # Total: avoid walk + get strikeout
                        return -walk_value + strikeout_cost
                    else:
                        # AB continues at (3, new_strikes)
                        count_change = (COUNT_RUN_VALUES.get((3, new_strikes), 0) -
                                       COUNT_RUN_VALUES.get((3, strikes), 0))
                        return -walk_value + count_change
                else:
                    # Regular count change (not walk situation)
                    # Count goes from (balls+1, strikes) to (balls, strikes+1)
                    old_count = (min(balls + 1, 3), strikes)
                    new_count = (balls, min(strikes + 1, 2))
                    
                    old_value = COUNT_RUN_VALUES.get(old_count, 0)
                    new_value = COUNT_RUN_VALUES.get(new_count, 0)
                    
                    return new_value - old_value
            
            return 0.0
        
        print("Calculating delta_re24 for each pitch...")
        self.df['delta_re24'] = self.df.apply(calculate_challenge_value, axis=1)
        
        # Expected value of challenge = P(overturn) × delta_re24
        self.df['expected_challenge_value'] = self.df['p_overturn'] * self.df['delta_re24']
        
        # Statistics
        print(f"\nDelta RE24 (if challenge succeeds):")
        print(f"  Mean:   {self.df['delta_re24'].mean():.4f}")
        print(f"  Std:    {self.df['delta_re24'].std():.4f}")
        print(f"  Min:    {self.df['delta_re24'].min():.4f}")
        print(f"  Max:    {self.df['delta_re24'].max():.4f}")
        
        # High-value situations
        high_value = self.df['delta_re24'].abs() > 0.1
        print(f"\nHigh-value situations (|delta_re24| > 0.1): {high_value.sum():,} ({high_value.mean()*100:.2f}%)")
        
        # Breakdown by situation
        print(f"\nBreakdown by call type:")
        for actual in ['strike', 'ball']:
            for true in ['strike', 'ball']:
                mask = (self.df['actual_call'] == actual) & (self.df['true_call'] == true)
                if mask.sum() > 0:
                    avg_delta = self.df.loc[mask, 'delta_re24'].mean()
                    print(f"  Called {actual}, was {true}: {mask.sum():,} pitches, avg delta={avg_delta:.4f}")
    
    def add_game_state_features(self):
        """Add game state features for Q-learning state space"""
        print("STEP 5: Engineering Game State Features")
        
        # Inning bucket
        self.df['inning_bucket'] = pd.cut(
            self.df['inning'],
            bins=[0, 3, 6, 9, 20],
            labels=['early', 'middle', 'late', 'extra']
        )
        
        # Score differential (from batting team perspective)
        self.df['score_diff'] = self.df['home_score'] - self.df['away_score']
        self.df['score_diff'] = np.where(
            self.df['inning_topbot'] == 'Top',
            -self.df['score_diff'],  # Away team batting
            self.df['score_diff']     # Home team batting
        )
        
        self.df['score_diff_bucket'] = pd.cut(
            self.df['score_diff'],
            bins=[-20, -3, -1, 1, 3, 20],
            labels=['down_big', 'down_small', 'close', 'up_small', 'up_big']
        )
        
        # Base-out state string
        self.df['base_out_state'] = (
            self.df['on_1b'].astype(str) + 
            self.df['on_2b'].astype(str) + 
            self.df['on_3b'].astype(str) + 
            '_' + 
            self.df['outs_when_up'].astype(str) + 'out'
        )
        
        # Leverage index (simplified)
        close_game = (self.df['score_diff'].abs() <= 2).astype(float) * 0.4
        late_inning = (self.df['inning'] >= 7).astype(float) * 0.3
        runners_on = ((self.df['on_1b'] + self.df['on_2b'] + self.df['on_3b']) > 0).astype(float) * 0.3
        self.df['leverage'] = close_game + late_inning + runners_on
        
        self.df['leverage_bucket'] = pd.cut(
            self.df['leverage'],
            bins=[-0.01, 0.3, 0.6, 1.01],
            labels=['low', 'medium', 'high']
        )
        
        # Count string
        self.df['count'] = self.df['balls'].astype(str) + '-' + self.df['strikes'].astype(str)
        
        print("Added features: inning_bucket, score_diff_bucket, leverage_bucket, count")
        print(f"  Leverage distribution: {self.df['leverage_bucket'].value_counts().to_dict()}")
    
    def clean_and_filter(self):
        """Final cleaning"""
        print("STEP 6: Final Cleaning")
        
        critical_cols = [
            'plate_x', 'plate_z', 'sz_top', 'sz_bot',
            'balls', 'strikes', 'outs_when_up',
            'on_1b', 'on_2b', 'on_3b', 'inning', 'game_pk'
        ]
        
        before = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        print(f"Removed {before - len(self.df):,} rows with missing data")
        
        # Validate game states
        valid = (
            (self.df['balls'] >= 0) & (self.df['balls'] <= 3) &
            (self.df['strikes'] >= 0) & (self.df['strikes'] <= 2) &
            (self.df['outs_when_up'] >= 0) & (self.df['outs_when_up'] <= 2)
        )
        before = len(self.df)
        self.df = self.df[valid]
        if before - len(self.df) > 0:
            print(f"Removed {before - len(self.df):,} invalid game states")
        
        print(f"Final dataset: {len(self.df):,} pitches")
    
    def create_train_val_test_splits(self):
        """Split by GAMES (not pitches) to prevent data leakage"""
        print("STEP 7: Train/Val/Test Splits")
        
        games = self.df['game_pk'].unique()
        print(f"Total games: {len(games)}")
        
        train_games, temp_games = train_test_split(games, test_size=0.3, random_state=42)
        val_games, test_games = train_test_split(temp_games, test_size=0.5, random_state=42)
        
        self.df['split'] = 'train'
        self.df.loc[self.df['game_pk'].isin(val_games), 'split'] = 'val'
        self.df.loc[self.df['game_pk'].isin(test_games), 'split'] = 'test'
        
        for split in ['train', 'val', 'test']:
            n = (self.df['split'] == split).sum()
            g = len(train_games) if split == 'train' else len(val_games) if split == 'val' else len(test_games)
            print(f"  {split}: {n:,} pitches from {g} games")
    
    def save_processed_data(self, output_dir='../data/processed'):
        """Save processed data"""
        print("STEP 8: Saving")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Full dataset
        full_path = f"{output_dir}/challengeable_pitches.csv"
        self.df.to_csv(full_path, index=False)
        print(f"Saved: {full_path}")
        
        # Splits
        for split in ['train', 'val', 'test']:
            split_df = self.df[self.df['split'] == split]
            split_df.to_csv(f"{output_dir}/{split}_data.csv", index=False)
        print(f"Saved train/val/test splits")
        
        # Summary stats
        summary = {
            'total_pitches': len(self.df),
            'total_games': self.df['game_pk'].nunique(),
            'incorrect_call_rate': f"{self.df['call_incorrect'].mean()*100:.2f}%",
            'avg_p_overturn': f"{self.df['p_overturn'].mean():.3f}",
            'avg_delta_re24': f"{self.df['delta_re24'].mean():.4f}",
            'max_delta_re24': f"{self.df['delta_re24'].max():.4f}",
            'min_delta_re24': f"{self.df['delta_re24'].min():.4f}",
            'high_value_pct': f"{(self.df['delta_re24'].abs() > 0.1).mean()*100:.2f}%",
        }
        
        with open(f"{output_dir}/summary_stats.txt", 'w') as f:
            f.write("ABS Challenge Strategy - Data Summary\n")
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved summary_stats.txt")
        
        # RE24 matrix for reference
        re24_df = pd.DataFrame([
            {'on_1b': k[0], 'on_2b': k[1], 'on_3b': k[2], 'outs': k[3], 're24': v}
            for k, v in RE24_MATRIX.items()
        ])
        re24_df.to_csv(f"{output_dir}/re24_matrix.csv", index=False)
        print(f"Saved re24_matrix.csv")
        
        return full_path


def main():
    """Main preprocessing pipeline"""
    
    # Input file (from data_collection.py)
    raw_path = '../data/raw/mlb_2024_full.csv'
    
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found!")
        print("Run data_collection.py first.")
        return
    
    # Run pipeline
    preprocessor = DataPreprocessor(raw_path)
    preprocessor.identify_challengeable_pitches()
    preprocessor.determine_true_call()
    preprocessor.calculate_overturn_probability()
    preprocessor.add_re24_features()
    preprocessor.add_game_state_features()
    preprocessor.clean_and_filter()
    preprocessor.create_train_val_test_splits()
    output = preprocessor.save_processed_data()
    
    print("PREPROCESSING COMPLETE!")
    
    print(f"\nOutput: {output}")
    print("\nNext: python q_learning.py")

if __name__ == "__main__":
    main()