"""
Autoencoder-based features for structural break detection.

Approach:
1. Train simple autoencoder on pre-break segment
2. Measure reconstruction error on both pre and post
3. Use error ratio/difference as feature

For time series, we use a sliding window to create 2D representation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


def split_pre_post(df_one_id) -> Tuple[np.ndarray, np.ndarray]:
    """Split a single series into pre-break and post-break segments."""
    if isinstance(df_one_id, pd.Series):
        df_one_id = df_one_id.to_frame()
    
    if 'period' in df_one_id.columns and 'value' in df_one_id.columns:
        pre_values = df_one_id[df_one_id['period'] == 0]['value'].values
        post_values = df_one_id[df_one_id['period'] == 1]['value'].values
    elif 'period' in df_one_id.index.names:
        if 'value' in df_one_id.columns:
            pre_values = df_one_id[df_one_id.index.get_level_values('period') == 0]['value'].values
            post_values = df_one_id[df_one_id.index.get_level_values('period') == 1]['value'].values
        else:
            df_reset = df_one_id.reset_index()
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
    else:
        df_reset = df_one_id.reset_index()
        if 'period' in df_reset.columns and 'value' in df_reset.columns:
            pre_values = df_reset[df_reset['period'] == 0]['value'].values
            post_values = df_reset[df_reset['period'] == 1]['value'].values
        else:
            raise ValueError(f"Cannot find 'period' and 'value' columns")
    
    return pre_values, post_values


def iter_series_data(df: pd.DataFrame):
    """Iterate over series in a DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        for series_id in df.index.get_level_values(0).unique():
            yield series_id, df.loc[series_id]
    else:
        for series_id, series_data in df.groupby('id', sort=False):
            yield series_id, series_data


class SimpleAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder.
    
    Input: flattened sliding windows
    Bottleneck: compressed representation
    Output: reconstructed windows
    """
    def __init__(self, input_dim, encoding_dim=8):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def create_windows(series, window_size=20, stride=5):
    """
    Create sliding windows from time series.
    
    Args:
        series: 1D array
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        2D array (n_windows, window_size)
    """
    if len(series) < window_size:
        # Pad if too short
        series = np.pad(series, (0, window_size - len(series)), mode='edge')
    
    windows = []
    for i in range(0, len(series) - window_size + 1, stride):
        windows.append(series[i:i+window_size])
    
    return np.array(windows)


def train_autoencoder(X, epochs=50, lr=0.001, batch_size=32, verbose=False):
    """
    Train autoencoder on windowed time series.
    
    Args:
        X: 2D array (n_windows, window_size)
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        verbose: Print training progress
    
    Returns:
        Trained autoencoder model
    """
    input_dim = X.shape[1]
    encoding_dim = max(4, input_dim // 4)  # Bottleneck 1/4 of input
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Convert to PyTorch
    X_tensor = torch.FloatTensor(X_norm)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = SimpleAutoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    
    model.eval()
    
    # Store normalization params
    model.X_mean = X_mean
    model.X_std = X_std
    
    return model


def compute_reconstruction_error(model, X):
    """
    Compute reconstruction error (MSE) for windows.
    
    Args:
        model: Trained autoencoder
        X: 2D array (n_windows, window_size)
    
    Returns:
        Mean reconstruction error
    """
    # Normalize with training stats
    X_norm = (X - model.X_mean) / model.X_std
    X_tensor = torch.FloatTensor(X_norm)
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()
    
    return mse.mean()


def compute_autoencoder_features(
    df: pd.DataFrame,
    window_size: int = 20,
    stride: int = 5,
    epochs: int = 30,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Compute autoencoder-based features for structural break detection.
    
    Features:
    - reconstruction_error_pre: Error on pre-break segment
    - reconstruction_error_post: Error on post-break segment
    - reconstruction_error_ratio: post / pre
    - reconstruction_error_diff: post - pre
    
    Args:
        df: Multi-index DataFrame with (id, period) index
        window_size: Sliding window size
        stride: Stride for sliding windows
        epochs: Training epochs
        verbose: Print progress
    
    Returns:
        DataFrame with autoencoder features
    """
    features = []
    ids = []
    
    if verbose:
        total = df.index.get_level_values(0).nunique() if isinstance(df.index, pd.MultiIndex) else len(df['id'].unique())
        print(f"Computing autoencoder features for {total} series...")
    
    for idx, (series_id, series_data) in enumerate(iter_series_data(df)):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} series")
        
        pre, post = split_pre_post(series_data)
        
        # Skip if too short
        if len(pre) < window_size or len(post) < window_size:
            features.append({
                'ae_recon_error_pre': 0,
                'ae_recon_error_post': 0,
                'ae_recon_error_ratio': 1,
                'ae_recon_error_diff': 0,
            })
            ids.append(series_id)
            continue
        
        try:
            # Create windows
            pre_windows = create_windows(pre, window_size, stride)
            post_windows = create_windows(post, window_size, stride)
            
            # Train autoencoder on pre-break data only
            model = train_autoencoder(
                pre_windows,
                epochs=epochs,
                lr=0.001,
                batch_size=min(32, len(pre_windows)),
                verbose=False
            )
            
            # Compute reconstruction errors
            error_pre = compute_reconstruction_error(model, pre_windows)
            error_post = compute_reconstruction_error(model, post_windows)
            
            # Features
            eps = 1e-8
            ratio = error_post / (error_pre + eps)
            diff = error_post - error_pre
            
            features.append({
                'ae_recon_error_pre': error_pre,
                'ae_recon_error_post': error_post,
                'ae_recon_error_ratio': ratio,
                'ae_recon_error_diff': diff,
            })
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed for series {series_id}: {e}")
            features.append({
                'ae_recon_error_pre': 0,
                'ae_recon_error_post': 0,
                'ae_recon_error_ratio': 1,
                'ae_recon_error_diff': 0,
            })
        
        ids.append(series_id)
    
    return pd.DataFrame(features, index=ids)


# Alternative: Extract embeddings as features
def compute_autoencoder_embeddings(
    df: pd.DataFrame,
    window_size: int = 20,
    stride: int = 10,
    encoding_dim: int = 8,
    epochs: int = 30,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Extract bottleneck embeddings as features.
    
    Returns embeddings for pre and post segments.
    """
    features = []
    
    series_ids = df.index.get_level_values(0).unique()
    
    if verbose:
        print(f"Computing autoencoder embeddings for {len(series_ids)} series...")
    
    for idx, series_id in enumerate(series_ids):
        if verbose and (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(series_ids)} series")
        
        series_data = df.loc[series_id]
        
        pre = series_data[series_data['period'] == 0]['value'].values
        post = series_data[series_data['period'] == 1]['value'].values
        
        if len(pre) < window_size or len(post) < window_size:
            # Zero embeddings if too short
            feat_dict = {}
            for i in range(encoding_dim):
                feat_dict[f'ae_emb_pre_{i}'] = 0
                feat_dict[f'ae_emb_post_{i}'] = 0
                feat_dict[f'ae_emb_diff_{i}'] = 0
            features.append(feat_dict)
            continue
        
        try:
            pre_windows = create_windows(pre, window_size, stride)
            post_windows = create_windows(post, window_size, stride)
            
            # Train on pre
            model = train_autoencoder(
                pre_windows,
                epochs=epochs,
                lr=0.001,
                batch_size=min(32, len(pre_windows)),
                verbose=False
            )
            
            # Extract embeddings
            pre_norm = (pre_windows - model.X_mean) / model.X_std
            post_norm = (post_windows - model.X_mean) / model.X_std
            
            with torch.no_grad():
                emb_pre = model.encode(torch.FloatTensor(pre_norm)).numpy().mean(axis=0)
                emb_post = model.encode(torch.FloatTensor(post_norm)).numpy().mean(axis=0)
            
            # Features
            feat_dict = {}
            for i in range(encoding_dim):
                feat_dict[f'ae_emb_pre_{i}'] = emb_pre[i]
                feat_dict[f'ae_emb_post_{i}'] = emb_post[i]
                feat_dict[f'ae_emb_diff_{i}'] = emb_post[i] - emb_pre[i]
            
            features.append(feat_dict)
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed for series {series_id}: {e}")
            feat_dict = {}
            for i in range(encoding_dim):
                feat_dict[f'ae_emb_pre_{i}'] = 0
                feat_dict[f'ae_emb_post_{i}'] = 0
                feat_dict[f'ae_emb_diff_{i}'] = 0
            features.append(feat_dict)
    
    return pd.DataFrame(features, index=series_ids)
