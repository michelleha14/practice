import numpy as np
import pandas as pd
import datetime

# Seed for reproducibility
np.random.seed(42)

# Parameters
num_subjects = 1200
days_per_subject = np.random.randint(5, 15, size=num_subjects) # 5 to 14 days per subject
points_per_day = 288 # 5-minute intervals in 24 hours

# Pre-generate subject-level metadata (A1c, FPG, 2-hr OGTT)
# A1c: mean ~5.7, std ~0.8 (e.g. 4.5 to 9.5)
# FPG (mg/dL): mean ~100, std ~20
# 2-hr OGTT (mg/dL): mean ~130, std ~35
a1c_base = np.random.normal(loc=5.7, scale=0.8, size=num_subjects)
a1c_base = np.clip(a1c_base, 4.5, 10.5)

fpg_base = a1c_base * 18.0 + np.random.normal(loc=0, scale=10, size=num_subjects) # rough correlation
ogtt_base = a1c_base * 25.0 + np.random.normal(loc=0, scale=15, size=num_subjects) # rough correlation

# We will generate data for each subject and collect in arrays/lists
subject_ids = []
timestamps_posixct = []
raw_glucose_list = []
sin_time_list = []
cos_time_list = []
raw_accel_list = []
raw_speed_list = []
a1c_list = []
fpg_list = []
ogtt_list = []

# Base start timestamp: 2026-11-01 00:00:00 UTC
base_start_time = datetime.datetime(2026, 11, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()

for subj_idx in range(num_subjects):
    subj_id = f"SUBJ_{subj_idx+1:04d}"
    n_days = days_per_subject[subj_idx]
    n_points = n_days * points_per_day
    
    # Timestamps in 5-min intervals (300 seconds)
    # Stagger subject start times slightly (0 to 12 hours)
    start_offset = np.random.uniform(0, 12 * 3600)
    t_seconds = base_start_time + start_offset + np.arange(n_points) * 300
    
    # Time of day features (sin / cos over 24-hour cycle)
    # Seconds into the day
    sec_in_day = t_seconds % 86400
    sin_time = np.sin(2 * np.pi * sec_in_day / 86400)
    cos_time = np.cos(2 * np.pi * sec_in_day / 86400)
    
    # Simulated Raw Glucose dynamics (AR(1) process with circadian meal peaks)
    # Base glucose centered around FPG/A1c relation
    mean_g = fpg_base[subj_idx]
    
    # Meal peaks roughly at 8:00 (28800s), 12:30 (45000s), 18:30 (66600s)
    meal_effect = (
        30.0 * np.exp(-((sec_in_day - 28800) / 3600)**2) +
        40.0 * np.exp(-((sec_in_day - 45000) / 3600)**2) +
        45.0 * np.exp(-((sec_in_day - 66600) / 3600)**2)
    )
    
    # Autoregressive noise for smooth continuous fluctuations
    noise = np.zeros(n_points)
    phi = 0.95
    for k in range(1, n_points):
        noise[k] = phi * noise[k-1] + np.random.normal(0, 3)
        
    raw_glucose = mean_g + meal_effect + noise
    raw_glucose = np.clip(raw_glucose, 40, 400) # Realistic physiological bounds
    
    # Simulated physical activity metrics: Acceleration and Speed
    # Non-negative, sparse bursts during daytime
    daytime_mask = (sec_in_day > 25200) & (sec_in_day < 79200) # 7 AM to 10 PM
    accel_raw = np.maximum(0, np.random.exponential(scale=0.1, size=n_points) * daytime_mask)
    speed_raw = np.maximum(0, np.random.exponential(scale=0.5, size=n_points) * daytime_mask)
    
    # Append to global lists
    subject_ids.extend([subj_id] * n_points)
    timestamps_posixct.extend(t_seconds)
    raw_glucose_list.extend(raw_glucose)
    sin_time_list.extend(sin_time)
    cos_time_list.extend(cos_time)
    raw_accel_list.extend(accel_raw)
    raw_speed_list.extend(speed_raw)
    a1c_list.extend([round(a1c_base[subj_idx], 2)] * n_points)
    fpg_list.extend([round(fpg_base[subj_idx], 1)] * n_points)
    ogtt_list.extend([round(ogtt_base[subj_idx], 1)] * n_points)

# Convert arrays to pandas DataFrame
df = pd.DataFrame({
    'subject_id': subject_ids,
    'timestamp_posixct': np.round(timestamps_posixct, 1),
    'raw_glucose': np.round(raw_glucose_list, 2),
    'sin_time': np.round(sin_time_list, 5),
    'cos_time': np.round(cos_time_list, 5),
    'raw_accel': np.round(raw_accel_list, 4),
    'raw_speed': np.round(raw_speed_list, 4),
    'a1c': a1c_list,
    'fpg': fpg_list,
    'ogtt_2hr': ogtt_list
})

# Global Z-score normalization across ALL subjects/rows
df['normalized_glucose'] = (df['raw_glucose'] - df['raw_glucose'].mean()) / df['raw_glucose'].std()
df['normalized_accel'] = (df['raw_accel'] - df['raw_accel'].mean()) / df['raw_accel'].std()
df['normalized_speed'] = (df['raw_speed'] - df['raw_speed'].mean()) / df['raw_speed'].std()

# Reorder columns logically
cols_order = [
    'subject_id',
    'timestamp_posixct',
    'raw_glucose',
    'normalized_glucose',
    'sin_time',
    'cos_time',
    'normalized_accel',
    'normalized_speed',
    'a1c',
    'fpg',
    'ogtt_2hr'
]
df = df[cols_order]

# Save to CSV
csv_filename = "pseudo_cgm_data_1200_subjects.csv"
df.to_csv(csv_filename, index=False)

print(f"Dataset generated successfully.")
print(f"Total Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")