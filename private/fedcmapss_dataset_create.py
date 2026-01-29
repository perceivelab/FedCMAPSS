import pandas as pd
import os
import re
import json
import numpy as np
from sklearn.cluster import KMeans

# Configuration
SOURCE_DATASET_DIR='private/dataset'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Will contain:
# - cmapss_processed_train_data.csv
# - cmapss_processed_test_data.csv
# - tasks.json
N_SPLITS_PER_TASK = 10
SUBSAMPLE_RATIO = 0.85

def extract_fd(file_path):
    """
    Extract FD from file path.
    """
    filename = os.path.basename(file_path)
    match = re.search(r'FD(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("FD number not found in filename")

def load_cmapss_data_file(file_path):
    """
    Load single data file.
    """
    # Extract FD
    fd = extract_fd(file_path)
    # Read file
    df = pd.read_csv(file_path, sep='\\s+', header=None)
    # Add FD column at the beginning
    df.insert(0, 'fd', fd)
    # Add header
    header = ['fd', 'unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df.columns = header
    return df

def load_cmapss_rul_file(file_path):
    """
    Load RUL file for test data.
    """
    # Extract FD
    fd = extract_fd(file_path)
    # Read file
    df = pd.read_csv(file_path, sep='\\s+', header=None)
    # Add FD column
    df.insert(0, 'fd', fd)
    # Add unit column
    df.insert(1, 'unit', range(1, len(df) + 1))
    df.columns = ['fd', 'unit', 'test_rul']
    return df

def load_cmapss_data(dir_path):
    """
    Load all data as data frames. Add the RUL column to each cycle.
    """
    # Check required files
    required_files = [
        'train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt',
        'test_FD001.txt', 'test_FD002.txt', 'test_FD003.txt', 'test_FD004.txt',
        'RUL_FD001.txt', 'RUL_FD002.txt', 'RUL_FD003.txt', 'RUL_FD004.txt'
    ]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(dir_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    # Read training data
    train_data_FD001 = load_cmapss_data_file(os.path.join(dir_path, 'train_FD001.txt'))
    train_data_FD002 = load_cmapss_data_file(os.path.join(dir_path, 'train_FD002.txt'))
    train_data_FD003 = load_cmapss_data_file(os.path.join(dir_path, 'train_FD003.txt'))
    train_data_FD004 = load_cmapss_data_file(os.path.join(dir_path, 'train_FD004.txt'))
    # Read test data
    test_data_FD001 = load_cmapss_data_file(os.path.join(dir_path, 'test_FD001.txt'))
    test_data_FD002 = load_cmapss_data_file(os.path.join(dir_path, 'test_FD002.txt'))
    test_data_FD003 = load_cmapss_data_file(os.path.join(dir_path, 'test_FD003.txt'))
    test_data_FD004 = load_cmapss_data_file(os.path.join(dir_path, 'test_FD004.txt'))
    # Add condition column set to -1
    train_data_FD001['condition'] = -1
    train_data_FD002['condition'] = -1
    train_data_FD003['condition'] = -1
    train_data_FD004['condition'] = -1
    test_data_FD001['condition'] = -1
    test_data_FD002['condition'] = -1
    test_data_FD003['condition'] = -1
    test_data_FD004['condition'] = -1
    # Cluster conditions from FD004 training data
    settings_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    train_data_FD004['condition'] = kmeans.fit_predict(train_data_FD004[settings_cols])
    test_data_FD004['condition'] = kmeans.predict(test_data_FD004[settings_cols])
    # Add RUL to training data
    train_data_FD001['rul'] = train_data_FD001.groupby('unit')['cycle'].transform('max') - train_data_FD001['cycle']
    train_data_FD002['rul'] = train_data_FD002.groupby('unit')['cycle'].transform('max') - train_data_FD002['cycle']
    train_data_FD003['rul'] = train_data_FD003.groupby('unit')['cycle'].transform('max') - train_data_FD003['cycle']
    train_data_FD004['rul'] = train_data_FD004.groupby('unit')['cycle'].transform('max') - train_data_FD004['cycle']
    # Read test RUL data
    test_rul_FD001 = load_cmapss_rul_file(os.path.join(dir_path, 'RUL_FD001.txt'))
    test_rul_FD002 = load_cmapss_rul_file(os.path.join(dir_path, 'RUL_FD002.txt'))
    test_rul_FD003 = load_cmapss_rul_file(os.path.join(dir_path, 'RUL_FD003.txt'))
    test_rul_FD004 = load_cmapss_rul_file(os.path.join(dir_path, 'RUL_FD004.txt'))
    # Join test data with test RUL based on fd and unit
    test_data_FD001 = pd.merge(test_data_FD001, test_rul_FD001, on=['fd', 'unit'])
    test_data_FD002 = pd.merge(test_data_FD002, test_rul_FD002, on=['fd', 'unit'])
    test_data_FD003 = pd.merge(test_data_FD003, test_rul_FD003, on=['fd', 'unit'])
    test_data_FD004 = pd.merge(test_data_FD004, test_rul_FD004, on=['fd', 'unit'])
    # Add RUL to test data
    test_data_FD001['rul'] = test_data_FD001.groupby('unit')['cycle'].transform('max') - test_data_FD001['cycle'] + test_data_FD001['test_rul']
    test_data_FD002['rul'] = test_data_FD002.groupby('unit')['cycle'].transform('max') - test_data_FD002['cycle'] + test_data_FD002['test_rul']
    test_data_FD003['rul'] = test_data_FD003.groupby('unit')['cycle'].transform('max') - test_data_FD003['cycle'] + test_data_FD003['test_rul']
    test_data_FD004['rul'] = test_data_FD004.groupby('unit')['cycle'].transform('max') - test_data_FD004['cycle'] + test_data_FD004['test_rul']
    # Drop test_rul column
    test_data_FD001.drop(columns=['test_rul'], inplace=True)
    test_data_FD002.drop(columns=['test_rul'], inplace=True)
    test_data_FD003.drop(columns=['test_rul'], inplace=True)
    test_data_FD004.drop(columns=['test_rul'], inplace=True)
    # Concatenate data
    train_data = pd.concat([train_data_FD001, train_data_FD002, train_data_FD003, train_data_FD004], ignore_index=True)
    test_data = pd.concat([test_data_FD001, test_data_FD002, test_data_FD003, test_data_FD004], ignore_index=True)
    return train_data, test_data
    
# Initialize splits
tasks = {}

# Load data
train_data, test_data = load_cmapss_data(SOURCE_DATASET_DIR)
# Save data
train_data.to_csv(f'{OUTPUT_DIR}/cmapss_processed_train_data.csv', index=False)
test_data.to_csv(f'{OUTPUT_DIR}/cmapss_processed_test_data.csv', index=False)

# Task A
TASK_ID = 'A'
N_CLIENTS = 10
FD = 1
TEST_FULL_SAMPLES_PER_CLIENT = 1
# Filter data for Task A (FD001)
train_df = train_data[train_data['fd'] == FD]
test_df = test_data[test_data['fd'] == FD]
# Get unique units
train_units = np.unique(train_df['unit'])
test_units = np.unique(test_df['unit'])
# Initialize splits
task_splits = {}
# Create splits
for split_id in range(N_SPLITS_PER_TASK):
    # Shuffle units
    rng = np.random.default_rng(seed=42 + split_id)
    rng.shuffle(train_units)
    rng.shuffle(test_units)
    # Split units among clients
    train_unit_splits = np.array_split(train_units, N_CLIENTS)
    test_full_unit_splits = [units[:TEST_FULL_SAMPLES_PER_CLIENT] for units in train_unit_splits]
    train_unit_splits = [units[TEST_FULL_SAMPLES_PER_CLIENT:] for units in train_unit_splits]
    test_unit_splits = np.array_split(test_units, N_CLIENTS)
    # Prepare split dictionary for saving (associate (fd, unit, condition) tuples to each client)
    task_splits[split_id] = {
        'train': {client_id: [(FD, int(unit), -1) for unit in train_unit_splits[client_id]] for client_id in range(N_CLIENTS)},
        'test_full': {client_id: [(FD, int(unit), -1) for unit in test_full_unit_splits[client_id]] for client_id in range(N_CLIENTS)},
        'test': {client_id: [(FD, int(unit), -1) for unit in test_unit_splits[client_id]] for client_id in range(N_CLIENTS)}
    }
# Add to tasks
tasks[TASK_ID] = task_splits

# Task B
TASK_ID = 'B'
N_CLIENTS = 4
# Initialize splits
task_splits = {}
# Create splits
for split_id in range(N_SPLITS_PER_TASK):
    # Initialize split dictionary
    split_dict = {'train': {}, 'test': {}, 'test_full': {}}
    # Create splits for each client
    for client_id in range(N_CLIENTS):
        # Map client_id to FD (0 -> FD1, 1 -> FD2, etc.)
        fd = client_id + 1
        # Filter data for this client
        client_train_units = np.unique(train_data[train_data['fd'] == fd]['unit'])
        client_test_units = np.unique(test_data[test_data['fd'] == fd]['unit'])
        # Shuffle units
        rng = np.random.default_rng(seed=42 + split_id*100 + client_id) 
        units_to_shuffle = np.array(client_train_units)
        rng.shuffle(units_to_shuffle)
        # Subsample
        selected_train_units = units_to_shuffle[:int(len(units_to_shuffle)*SUBSAMPLE_RATIO)]
        test_full_samples = units_to_shuffle[len(selected_train_units):]
        # Assign to client
        split_dict['train'][client_id] = [(fd, int(unit), -1) for unit in selected_train_units]
        split_dict['test_full'][client_id] = [(fd, int(unit), -1) for unit in test_full_samples]
        split_dict['test'][client_id] = [(fd, int(unit), -1) for unit in client_test_units]
    # Save split
    task_splits[split_id] = split_dict
# Add to tasks
tasks[TASK_ID] = task_splits

# Task C
TASK_ID = 'C'
N_CLIENTS = 10
FD = 4
# Filter data for Task C (FD004)
train_df = train_data[train_data['fd'] == FD]
test_df = test_data[test_data['fd'] == FD]
# Sort units by lifespan
# 1. Training Data: Lifespan = sequence length (max cycle)
train_lifespans = train_df.groupby('unit')['cycle'].max().sort_values()
sorted_train_units = train_lifespans.index.to_numpy()
# 2. Test Data: Lifespan = sequence length + RUL
grouped_test = test_df.groupby('unit')
test_lifespans = (grouped_test['cycle'].max() + grouped_test['rul'].first()).sort_values()
sorted_test_units = test_lifespans.index.to_numpy()
# Split sorted units into client blocks
client_train_blocks = np.array_split(sorted_train_units, N_CLIENTS)
client_test_blocks = np.array_split(sorted_test_units, N_CLIENTS)
# Initialize splits
task_splits = {}
# Create splits
for split_id in range(N_SPLITS_PER_TASK):
    # Initialize split dictionary
    split_dict = {'train': {}, 'test': {}, 'test_full': {}}
    # Create splits for each client
    for client_id in range(N_CLIENTS):
        # Get units allocated to this client (based on lifespan bucket)
        client_train_units = client_train_blocks[client_id]
        client_test_units = client_test_blocks[client_id]
        # Shuffle units
        rng = np.random.default_rng(seed=42 + split_id*100 + client_id)
        units_to_shuffle = np.array(client_train_units)
        rng.shuffle(units_to_shuffle)
        # Subsample
        selected_train_units = units_to_shuffle[:int(len(units_to_shuffle)*SUBSAMPLE_RATIO)]
        test_full_samples = units_to_shuffle[len(selected_train_units):]
        # Assign to client
        split_dict['train'][client_id] = [(FD, int(unit), -1) for unit in selected_train_units]
        split_dict['test_full'][client_id] = [(FD, int(unit), -1) for unit in test_full_samples]
        split_dict['test'][client_id] = [(FD, int(unit), -1) for unit in client_test_units]
    # Save split    
    task_splits[split_id] = split_dict
# Add to tasks
tasks[TASK_ID] = task_splits

# Task D
TASK_ID = 'D'
N_CLIENTS = 6
FD = 4
# Filter data for Task D (FD004)
train_df = train_data[train_data['fd'] == FD]
test_df = test_data[test_data['fd'] == FD]
# Get all units
train_units = np.unique(train_df['unit'])
test_units = np.unique(test_df['unit'])
# Initialize splits
task_splits = {}
# Create splits
for split_id in range(N_SPLITS_PER_TASK):
    # Initialize split dictionary
    split_dict = {'train': {}, 'test': {}, 'test_full': {}}
    # Create splits for each client
    for client_id in range(N_CLIENTS):
        # Set condition id
        condition_id = client_id
        # Shuffle units
        rng = np.random.default_rng(seed=42 + split_id*100 + client_id)
        units_to_shuffle = np.array(train_units)
        rng.shuffle(units_to_shuffle)
        # Subsample
        current_train_units = units_to_shuffle[:int(len(units_to_shuffle)*SUBSAMPLE_RATIO)]
        test_full_samples = units_to_shuffle[len(current_train_units):]
        # Assign to client
        split_dict['train'][client_id] = [(FD, int(unit), int(condition_id)) for unit in current_train_units]
        split_dict['test_full'][client_id] = [(FD, int(unit), int(condition_id)) for unit in test_full_samples]
        split_dict['test'][client_id] = [(FD, int(unit), int(condition_id)) for unit in test_units]
    # Save split    
    task_splits[split_id] = split_dict
# Add to tasks
tasks[TASK_ID] = task_splits

# Task E
TASK_ID = 'E'
FD = 1
# Filter data for Task E (FD001)
train_df = train_data[train_data['fd'] == FD]
test_df = test_data[test_data['fd'] == FD]
# Get all units (Assumes same units in train and test)
units = np.unique(train_df['unit'])
n_total_units = len(units)
# Determine N_CLIENTS based on subsample ratio of UNITS (Clients)
N_CLIENTS = int(n_total_units * SUBSAMPLE_RATIO) 
# Initialize splits
task_splits = {}
# Create splits
for split_id in range(N_SPLITS_PER_TASK):
    split_dict = {'train': {}, 'test': {}, 'test_full': {}}
    # Shuffle units to determine which are active clients and which are unused
    rng = np.random.default_rng(seed=42 + split_id*100)
    shuffled_units = np.array(units)
    rng.shuffle(shuffled_units)
    # Select active and unused units
    active_units = shuffled_units[:N_CLIENTS]
    unused_units = shuffled_units[N_CLIENTS:]
    # Create splits for each client
    for client_id in range(N_CLIENTS):
        # Client assigned exactly one unit
        unit = active_units[client_id]
        # Train and test on that specific unit
        split_dict['train'][client_id] = [(FD, int(unit), -1)]
        split_dict['test'][client_id] = [(FD, int(unit), -1)]
        # Assign unused units as test_full until we run out
        limit_idx = len(unused_units)
        if client_id < limit_idx:
            unused_unit = unused_units[client_id]
            split_dict['test_full'][client_id] = [(FD, int(unused_unit), -1)]
        else:
            split_dict['test_full'][client_id] = []
    # Save split
    task_splits[split_id] = split_dict
# Add to tasks
tasks[TASK_ID] = task_splits

# Save splits to file
with open(f'{OUTPUT_DIR}/tasks.json', 'w') as f:
    json.dump(tasks, f) #, indent=4)