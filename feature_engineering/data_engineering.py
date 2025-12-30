"""
Data Engineering - Feature Extraction for S-FFSD Dataset
=========================================================

This file contains functions to engineer features from the raw S-FFSD 
transaction data. It generates:

1. 2D Feature Matrices (for STAGN):
    - Shape: (N, 5, 8) - N transactions, 5 features, 8 time windows
    - Features: AvgAmount, TotalAmount, BiasAmount, Count, TradingEntropy
    - Time windows: [1, 3, 5, 10, 20, 50, 100, 500] transactions
    
2. 127 Temporal Features (for GTAN/RGTAN):
    - 15 time windows × 8 statistics = 120 features + raw fields = 127

Key Functions:
    - span_data_2d(): Generate 2D feature matrices for STAGN
    - span_data_3d(): Generate 3D feature matrices (spatial-temporal)
    - data_engineer_benchmark(): Generate 127 temporal features for GTAN/RGTAN
    - featmap_gen(): Per-source feature generation
    - calcu_trading_entropy(): Compute transaction type entropy
"""

import pandas as pd
import numpy as np
from math import isnan
import multiprocessing as mp
import sys
from tqdm import tqdm


def data_engineer_example(data_dir):
    """
    Example data engineering function for credit card fraud detection
    
    This is a reference implementation showing how to generate features
    from credit card transaction data with multiple time windows.
    
    Args:
        data_dir (str): Path to the CSV data file
        
    Returns:
        0 on success (saves numpy files)
    """
    data = pd.read_csv(data_dir)
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])

    # Define time windows for feature aggregation
    time_span = []
    # 1min, 1hr, 1day, 2days, 1month, 3months, 6months, 1year
    for i in [60, 3600, 86400, 172800, 2628000, 7884000, 15768000, 31536000]:
        time_span.append(pd.Timedelta(seconds=i))

    train = []
    Oct = []
    Nov = []
    Dec = []

    start_time = "2015/1/1 00:00"

    # Process each transaction
    for i in data.iterrows():
        data2 = []
        temp_data = data[data['card_id'] == i[1]['card_id']]
        temp_county_id = i[1]['loc_cty']
        temp_merch_id = i[1]['loc_merch']
        temp_time = i[1]['time_stamp']
        temp_label = i[1]['is_fraud']
        a_grant = i[1]['amt_grant']
        a_purch = i[1]['amt_purch']
        
        # For each location, compute aggregated features
        for loc in data['loc_cty'].unique():
            data1 = []
            if (loc in temp_data['loc_cty'].unique()):
                card_tuple = temp_data['loc_cty'] == loc
                single_loc_card_data = temp_data[card_tuple]
                time_list = single_loc_card_data['time_stamp']
                
                # Compute features for each time window
                for length in time_span:
                    lowbound = (time_list >= (temp_time - length))
                    upbound = (time_list <= temp_time)
                    correct_data = single_loc_card_data[lowbound & upbound]
                    
                    # Aggregate statistics
                    Avg_grt_amt = correct_data['amt_grant'].mean()
                    Totl_grt_amt = correct_data['amt_grant'].sum()
                    Avg_pur_amt = correct_data['amt_purch'].mean()
                    Totl_pur_amt = correct_data['amt_purch'].sum()
                    Num = correct_data['amt_grant'].count()
                    
                    if (isnan(Avg_grt_amt)):
                        Avg_grt_amt = 0
                    if (isnan(Avg_pur_amt)):
                        Avg_pur_amt = 0
                    data1.append([a_grant, Avg_grt_amt, Totl_grt_amt,
                                  a_purch, Avg_pur_amt, Totl_pur_amt, Num])
            else:
                for length in time_span:
                    data1.append([0, 0, 0, 0, 0, 0, 0])
            data2.append(data1)
            
        # Split into train/test by month
        if (temp_time > pd.to_datetime(start_time)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
                train.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
                Oct.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
                Nov.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=12 * 2628000)):
                Dec.append([temp_label, np.array(data2)])
                
    np.save(file='train', arr=train)
    np.save(file='Oct', arr=Oct)
    np.save(file='Nov', arr=Nov)
    np.save(file='Dec', arr=Dec)
    return 0


def featmap_gen(tmp_card, tmp_df=None):
    """
    Generate temporal features for a single Source account
    
    This function is called in parallel for each Source account.
    It generates aggregate statistics over different time windows
    for each transaction from that account.
    
    Time Windows: [5, 20] transactions (can be extended)
    
    Features generated per time window:
        1. trans_at_avg_T: Average transaction amount
        2. trans_at_totl_T: Total transaction amount
        3. trans_at_std_T: Standard deviation of amounts
        4. trans_at_bias_T: Current amount - average amount
        5. trans_at_num_T: Number of transactions
        6. trans_target_num_T: Number of unique targets
        7. trans_location_num_T: Number of unique locations
        8. trans_type_num_T: Number of unique transaction types
    
    Args:
        tmp_card: Source account identifier
        tmp_df (DataFrame): Transactions for this source account
        
    Returns:
        DataFrame: Augmented transaction data with engineered features
    """
    # Time windows (in number of transactions)
    time_span = [5, 20]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    
    for trans_idx, trans_feat in tmp_df.iterrows():
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        
        # Generate features for each time window
        for length, tname in zip(time_span, time_name):
            # Select transactions within time window
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            
            # ============ AMOUNT STATISTICS ============
            # Average amount in window
            new_df['trans_at_avg_{}'.format(tname)] = correct_data['Amount'].mean()
            # Total amount in window
            new_df['trans_at_totl_{}'.format(tname)] = correct_data['Amount'].sum()
            # Standard deviation of amounts
            new_df['trans_at_std_{}'.format(tname)] = correct_data['Amount'].std()
            # Bias: how different is current amount from average
            new_df['trans_at_bias_{}'.format(tname)] = temp_amt - correct_data['Amount'].mean()
            
            # ============ COUNT STATISTICS ============
            # Number of transactions in window
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            # Number of unique targets
            new_df['trans_target_num_{}'.format(tname)] = len(correct_data.Target.unique())
            # Number of unique locations
            new_df['trans_location_num_{}'.format(tname)] = len(correct_data.Location.unique())
            # Number of unique transaction types
            new_df['trans_type_num_{}'.format(tname)] = len(correct_data.Type.unique())
            
        post_fe.append(new_df)
        
    return pd.DataFrame(post_fe)


def data_engineer_benchmark(feat_df):
    """
    Generate 127 temporal features for GTAN/RGTAN models
    
    This function uses multiprocessing to efficiently generate
    temporal aggregation features for all transactions.
    
    Process:
        1. Group transactions by Source account
        2. Parallel process each group with featmap_gen()
        3. Concatenate results and fill NaN with 0
    
    The final feature vector contains:
        - Raw fields: Time, Source, Target, Amount, Location, Type, Labels (7)
        - Aggregated features over time windows (120)
        - Total: 127 features
    
    Args:
        feat_df (DataFrame): Raw S-FFSD transaction data
        
    Returns:
        DataFrame: Feature-engineered data with 127 columns
    """
    # Use 4 parallel processes
    pool = mp.Pool(processes=4)
    
    # Prepare arguments for each Source account
    args_all = [(card_n, card_df) for card_n, card_df in feat_df.groupby("Source")]
    
    # Submit parallel jobs
    jobs = [pool.apply_async(featmap_gen, args=args) for args in args_all]
    
    # Collect results with progress display
    post_fe_df = []
    num_job = len(jobs)
    for i, job in enumerate(jobs):
        post_fe_df.append(job.get())
        sys.stdout.flush()
        sys.stdout.write("FE: {}/{}\r".format(i + 1, num_job))
        sys.stdout.flush()
        
    # Concatenate all results
    post_fe_df = pd.concat(post_fe_df)
    # Fill missing values with 0
    post_fe_df = post_fe_df.fillna(0.)
    
    return post_fe_df


def calcu_trading_entropy(data_2: pd.DataFrame) -> float:
    """
    Calculate trading entropy for a set of transactions
    
    Trading entropy measures the diversity of transaction types.
    High entropy = diverse transaction types (normal behavior)
    Low entropy = concentrated in few types (potentially suspicious)
    
    Formula: H = -Σ p_i * log(p_i)
    where p_i is the proportion of amount for transaction type i
    
    Args:
        data_2 (DataFrame): DataFrame with 'Amount' and 'Type' columns
        
    Returns:
        float: Entropy value (higher = more diverse)
    """
    # Handle empty dataframe
    if len(data_2) == 0:
        return 0

    # Calculate total amount per transaction type
    amounts = np.array([
        data_2[data_2['Type'] == type]['Amount'].sum()
        for type in data_2['Type'].unique()
    ])
    
    # Calculate proportions (add small epsilon to avoid log(0))
    proportions = amounts / amounts.sum() if amounts.sum() else np.ones_like(amounts)
    
    # Calculate entropy: -Σ p * log(p)
    ent = -np.array([
        proportion * np.log(1e-5 + proportion)
        for proportion in proportions
    ]).sum()
    
    return ent


def span_data_2d(
        data: pd.DataFrame,
        time_windows: list = [1, 3, 5, 10, 20, 50, 100, 500]
) -> np.ndarray:
    """
    Generate 2D feature matrices for STAGN model
    
    This function transforms each transaction into a 2D feature matrix
    suitable for the STAGN model which uses 2D CNN.
    
    Output Shape: (N, 5, 8)
        - N: Number of transactions (excluding unlabeled)
        - 5: Number of features per time window
        - 8: Number of time windows
    
    The 5 features per time window:
        1. AvgAmountT: Average transaction amount in window
        2. TotalAmountT: Total transaction amount in window
        3. BiasAmountT: Current amount - average (detects unusual amounts)
        4. NumberT: Transaction count in window
        5. TradingEntropyT: Change in transaction type entropy
    
    The 8 time windows (in number of prior transactions):
        [1, 3, 5, 10, 20, 50, 100, 500]
    
    Args:
        data (DataFrame): Raw S-FFSD transaction data
        time_windows (list): List of time window sizes
        
    Returns:
        tuple: (features, labels)
            - features: ndarray of shape (N, 5, 8)
            - labels: ndarray of shape (N,) with values 0 or 1
    """
    # Filter out unlabeled transactions (label == 2)
    data = data[data['Labels'] != 2]

    nume_feature_ret, label_ret = [], []
    
    # Process each transaction
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        feature_of_one_record = []

        # Generate features for each time window
        for time_span in time_windows:
            feature_of_one_timestamp = []
            
            # Get previous transactions (within time window)
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(row_idx - time_span):row_idx + 1, :]
            
            # Filter to same source account
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # ============ FEATURE 1: Average Amount ============
            feature_of_one_timestamp.append(
                prev_records['Amount'].sum() / time_span)
            
            # ============ FEATURE 2: Total Amount ============
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            
            # ============ FEATURE 3: Bias Amount ============
            # How different is current amount from recent average
            feature_of_one_timestamp.append(
                record['Amount'] - feature_of_one_timestamp[0])
            
            # ============ FEATURE 4: Transaction Count ============
            feature_of_one_timestamp.append(len(prev_records))

            # ============ FEATURE 5: Trading Entropy Change ============
            # Measures how the addition of current transaction changes entropy
            # Large change = unusual transaction type pattern
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)

            feature_of_one_record.append(feature_of_one_timestamp)

        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    # Reshape: (N, 8, 5) → (N, 5, 8) 
    # Puts features as rows, time windows as columns
    nume_feature_ret = np.array(nume_feature_ret).transpose(0, 2, 1)

    # Sanity check
    assert nume_feature_ret.shape == (
        len(data), 5, len(time_windows)), "output shape invalid."

    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)


def span_data_3d(
        data: pd.DataFrame,
        time_windows=None,
        spatio_windows=None,
) -> np.ndarray:
    """
    Generate 3D feature matrices for STAN model
    
    This extends 2D features with spatial (location) windows.
    
    Output Shape: (N, 8, 5, 5)
        - N: Number of transactions
        - 8: Time windows
        - 5: Spatial windows (location ranges)
        - 5: Features per window
    
    The spatial windows group transactions by location proximity.
    
    Args:
        data (DataFrame): Raw S-FFSD transaction data
        time_windows (list): Time window sizes
        spatio_windows (list): Spatial window sizes
        
    Returns:
        tuple: (features, labels)
            - features: ndarray of shape (N, 8, 5, 5)
            - labels: ndarray of shape (N,)
    """
    # Default windows
    if time_windows is None:
        time_windows = [1, 3, 5, 10, 20, 50, 100, 500]
    if spatio_windows is None:
        spatio_windows = [1, 2, 3, 4, 5]
        
    # Filter unlabeled data
    data = data[data['Labels'] != 2]
    
    # Encode location into 5 spatial zones
    data['Location'] = data['Location'].apply(lambda x: int(x.split('L')[1]))
    data['Location'] = data['Location'].apply(lambda x: 1 if x == 100 else x)
    data['Location'] = data['Location'].apply(lambda x: 2 if 102 >= x > 100 else x)
    data['Location'] = data['Location'].apply(lambda x: 3 if 110 >= x > 102 else x)
    data['Location'] = data['Location'].apply(lambda x: 4 if 140 >= x > 110 else x)
    data['Location'] = data['Location'].apply(lambda x: 5 if x > 140 else x)

    nume_feature_ret, label_ret = [], []
    
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        location = int(record['Location'])
        feature_of_one_record = []
        
        # For each time window
        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # For each spatial window
            for spatio_span in spatio_windows:
                feature_of_one_spatio_stamp = []
                
                # Filter by spatial proximity
                one_spatio_records = prev_records[prev_records['Location'] > location - spatio_span]
                one_spatio_records = one_spatio_records[one_spatio_records['Location'] < location + spatio_span]

                # Same 5 features as 2D
                feature_of_one_spatio_stamp.append(
                    one_spatio_records['Amount'].sum() / time_span)
                feature_of_one_spatio_stamp.append(one_spatio_records['Amount'].sum())
                feature_of_one_spatio_stamp.append(
                    record['Amount'] - feature_of_one_spatio_stamp[0])
                feature_of_one_spatio_stamp.append(len(one_spatio_records))

                old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
                new_ent = calcu_trading_entropy(prev_and_now_records[['Amount', 'Type']])
                feature_of_one_spatio_stamp.append(old_ent - new_ent)

                feature_of_one_timestamp.append(feature_of_one_spatio_stamp)
            feature_of_one_record.append(feature_of_one_timestamp)
            
        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    nume_feature_ret = np.array(nume_feature_ret)
    print(nume_feature_ret.shape)
    
    # Sanity check
    assert nume_feature_ret.shape == (
        len(data), len(time_windows), len(spatio_windows), 5), "output shape invalid."

    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)
