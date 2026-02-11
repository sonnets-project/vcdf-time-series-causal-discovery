"""
vcdf_framework.py

This module implements a generic Validated Consensus Driven Framework (VCDF) for causal discovery methods.
The framework can be used with different base methods (like VAR-LiNGAM, PCMCI, etc.) to improve
the reliability of causal discovery in time series data.

Includes both the base VCDF implementation and grid search functionality.

Author: Gene Yu
Date: August 2024
"""

import numpy as np
from sklearn.model_selection import KFold
from itertools import product
from src.causal_matrix_evaluation import evaluate_causal_matrices

def time_series_kfold_split(data, n_splits):
    """
    時間序列感知的k-fold分割，每個fold的training set都是連續的
    
    例如 k=7 時：
    - fold 0: validation=[0], training=[1,2,3,4,5,6] (右邊6個)
    - fold 1: validation=[1], training=[2,3,4,5,6] (右邊5個)  
    - fold 2: validation=[2], training=[3,4,5,6] (右邊4個)
    - fold 3: validation=[3], training=[0,1,2] (左邊3個，因為右邊只有3個)
    - fold 4: validation=[4], training=[0,1,2,3] (左邊4個)
    - fold 5: validation=[5], training=[0,1,2,3,4] (左邊5個)
    - fold 6: validation=[6], training=[0,1,2,3,4,5] (左邊6個)
    
    Parameters:
    -----------
    data : numpy.ndarray
        時間序列數據
    n_splits : int
        k-fold的k值
        
    Returns:
    --------
    list of tuples: [(train_indices, val_indices), ...]
    """
    n_samples = len(data)
    fold_size = n_samples // n_splits
    
    splits = []
    for i in range(n_splits):
        # validation set是連續的
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_splits - 1 else n_samples
        val_indices = list(range(val_start, val_end))
        
        # 計算左邊和右邊的連續segment長度
        left_size = val_start  # 左邊可用的連續樣本數
        right_size = n_samples - val_end  # 右邊可用的連續樣本數
        
        # 選擇較長的連續segment作為training set
        if left_size >= right_size:
            # 使用左邊的連續segment
            train_start = 0
            train_end = val_start
            train_indices = list(range(train_start, train_end))
        else:
            # 使用右邊的連續segment
            train_start = val_end
            train_end = n_samples
            train_indices = list(range(train_start, train_end))
        
        splits.append((train_indices, val_indices))
    
    return splits

def run_vcdf(data, base_method, n_splits=5, consistency_threshold=0.4, 
            variability_threshold=0.4, adjustment_weight=0, **method_params):
    """
    Generic VCDF implementation that can be used with different base methods.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input time series data
    base_method : function
        Base causal discovery method to use (e.g., run_varlingam, run_pcmci)
    n_splits : int
        Number of splits for k-fold validation
    consistency_threshold : float
        Threshold for consistency check
    variability_threshold : float
        Threshold for variability check
    adjustment_weight : float
        Weight for adjusting the initial estimates
    method_params : dict
        Additional parameters specific to the base method
        
    Returns:
    --------
    list
        List of validated adjacency matrices
    """
    # Convert DataFrame to numpy array if needed
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = data
    
    # Initial fit with all data
    initial_matrices = base_method(data, **method_params)
    n_lags = len(initial_matrices)
    n_vars = initial_matrices[0].shape[0]

    # Use standard sklearn KFold splitting (random sampling)
    kf = KFold(n_splits=n_splits, shuffle=False)
    splits = list(kf.split(data_array))
    
    # Alternative: Use time series aware k-fold splitting (preserves temporal structure)
    # splits = time_series_kfold_split(data_array, n_splits)
    
    all_matrices = []
    
    for train_indices, val_indices in splits:
        train_data = data_array[train_indices]
        # Convert back to DataFrame if original was DataFrame
        if hasattr(data, 'columns'):
            import pandas as pd
            train_data = pd.DataFrame(train_data, columns=data.columns)
        cv_matrices = base_method(train_data, **method_params)
        padded_matrices = pad_or_truncate_matrices(cv_matrices, n_lags-1, n_vars)
        all_matrices.append(padded_matrices)
    
    # Validation and adjustment
    validated_matrices = validate_and_adjust_matrices(
        initial_matrices, all_matrices, 
        consistency_threshold, variability_threshold, 
        adjustment_weight
    )
    
    return validated_matrices

def pad_or_truncate_matrices(matrices, target_n_lags, n_vars):
    """Pad or truncate matrices to match target number of lags"""
    current_n_lags = len(matrices) - 1
    
    if current_n_lags < target_n_lags:
        padding = [np.zeros((n_vars, n_vars)) for _ in range(target_n_lags - current_n_lags)]
        matrices = list(matrices) + padding
    elif current_n_lags > target_n_lags:
        matrices = matrices[:target_n_lags + 1]
    
    return matrices

def remove_outliers(data):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def validate_and_adjust_matrices(initial_matrices, all_matrices, consistency_threshold, 
                               variability_threshold, adjustment_weight):
    """Validate and adjust matrices based on results from k-folds"""
    n_lags = len(initial_matrices)
    n_vars = initial_matrices[0].shape[0]
    
    validated_matrices = []
    for lag in range(n_lags):
        validated_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                initial_value = initial_matrices[lag][i, j]
                fold_values = [m[lag][i, j] for m in all_matrices]
                
                # Check consistency
                consistent_count = sum(1 for v in fold_values if np.sign(v) == np.sign(initial_value))
                consistency = consistent_count / len(fold_values)
                
                # Check variability
                variability = np.std(fold_values) / (np.abs(initial_value) + 1e-8)
                
                if consistency > consistency_threshold and variability < variability_threshold:
                    cleaned_values = remove_outliers(fold_values)
                    mean_value = np.mean(cleaned_values) if cleaned_values else initial_value
                    adjusted_value = (1 - adjustment_weight) * initial_value + adjustment_weight * mean_value
                    validated_matrix[i, j] = adjusted_value
                
        validated_matrices.append(validated_matrix)
    
    return validated_matrices

def grid_search_vcdf(data, true_matrices, base_method, param_grid=None, method_params=None):
    """
    Perform grid search for VCDF parameters with any base causal discovery method.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Input time series data
    true_matrices : list
        True adjacency matrices for evaluation
    base_method : function
        Base causal discovery method (e.g., run_varlingam, run_pcmci)
    param_grid : dict, optional
        Grid of VCDF parameters to search. If None, uses default grid
    method_params : dict, optional
        Additional parameters for the base method
        
    Returns:
    --------
    dict
        Contains best parameters, best score, and best matrices
    """
    if param_grid is None:
        param_grid = {
            'n_splits': [3, 5, 7],
            'consistency_threshold': [0.3, 0.5, 0.7],
            'variability_threshold': [0.3, 0.5, 0.7],
            'adjustment_weight': [0.0, 0.1, 0.2]
        }
    
    if method_params is None:
        method_params = {}

    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())
    
    best_score = float('-inf')
    best_params = None
    best_matrices = None
    
    # Try each parameter combination
    for params in param_combinations:
        current_params = dict(zip(param_keys, params))
        
        # Run VCDF with current parameters
        matrices = run_vcdf(
            data=data,
            base_method=base_method,
            **current_params,
            **method_params
        )
        
        # Evaluate results
        scores = evaluate_causal_matrices(true_matrices, matrices)
        current_score = scores['f1']
        
        # Update best results if current score is better
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_matrices = matrices
            
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }