"""
Feature Extraction Module for SEED EEG Data

This module implements various feature extraction methods for EEG emotion recognition:
- Power Spectral Density (PSD)
- Differential Entropy (DE)
- Asymmetry features (DASM, RASM, DCAU)
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EEGFeatureExtractor:
    """
    Feature extractor for EEG emotion recognition.
    
    Implements multiple feature extraction methods including spectral,
    entropy-based, and asymmetry features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.sampling_rate = config['data']['sampling_rate']
        self.frequency_bands = config['data']['frequency_bands']
        self.window_length = config['features']['window_length']
        self.overlap = config['features']['overlap']
        
        logger.info("Initialized EEG feature extractor")
    
    def extract_frequency_bands(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract frequency band-specific data using bandpass filtering.
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with band names as keys and filtered data as values
        """
        band_data = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Design bandpass filter
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Handle edge cases
            if low <= 0:
                low = 0.01
            if high >= 1:
                high = 0.99
            
            try:
                b, a = signal.butter(4, [low, high], btype='band')
                
                # Apply filter to each trial and channel
                filtered_data = np.zeros_like(data)
                for trial in range(data.shape[0]):
                    for ch in range(data.shape[1]):
                        filtered_data[trial, ch, :] = signal.filtfilt(b, a, data[trial, ch, :])
                
                band_data[band_name] = filtered_data
                logger.debug(f"Extracted {band_name} band ({low_freq}-{high_freq} Hz)")
                
            except Exception as e:
                logger.error(f"Error filtering {band_name} band: {e}")
                band_data[band_name] = data  # Fallback to original data
        
        return band_data
    
    def compute_psd_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute Power Spectral Density (PSD) features.
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            np.ndarray: PSD features with shape (n_trials, n_channels * n_bands)
        """
        n_trials, n_channels, n_timepoints = data.shape
        band_data = self.extract_frequency_bands(data)
        
        # Compute PSD for each frequency band
        psd_features = []
        
        for trial in range(n_trials):
            trial_features = []
            
            for ch in range(n_channels):
                channel_features = []
                
                for band_name, band_signal in band_data.items():
                    # Compute PSD using Welch's method
                    freqs, psd = signal.welch(
                        band_signal[trial, ch, :],
                        fs=self.sampling_rate,
                        nperseg=min(256, n_timepoints//4)
                    )
                    
                    # Take mean power in the frequency band
                    low_freq, high_freq = self.frequency_bands[band_name]
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    mean_power = np.mean(psd[freq_mask]) if np.any(freq_mask) else 0.0
                    
                    channel_features.append(mean_power)
                
                trial_features.extend(channel_features)
            
            psd_features.append(trial_features)
        
        psd_features = np.array(psd_features)
        # logger.info(f"Computed PSD features: shape {psd_features.shape}")
        
        return psd_features
    
    def compute_de_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute Differential Entropy (DE) features.
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            
        Returns:
            np.ndarray: DE features with shape (n_trials, n_channels * n_bands)
        """
        n_trials, n_channels, n_timepoints = data.shape
        band_data = self.extract_frequency_bands(data)
        
        de_features = []
        
        for trial in range(n_trials):
            trial_features = []
            
            for ch in range(n_channels):
                channel_features = []
                
                for band_name, band_signal in band_data.items():
                    # Compute differential entropy
                    signal_data = band_signal[trial, ch, :]
                    
                    # Remove DC component
                    signal_data = signal_data - np.mean(signal_data)
                    
                    # Compute variance (related to differential entropy for Gaussian signals)
                    variance = np.var(signal_data)
                    
                    # Differential entropy for Gaussian: 0.5 * log(2 * pi * e * variance)
                    de_value = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
                    
                    channel_features.append(de_value)
                
                trial_features.extend(channel_features)
            
            de_features.append(trial_features)
        
        de_features = np.array(de_features)
        # logger.info(f"Computed DE features: shape {de_features.shape}")
        
        return de_features
    
    def compute_asymmetry_features(self, data: np.ndarray, feature_type: str = 'dasm') -> np.ndarray:
        """
        Compute asymmetry features (DASM, RASM, DCAU).
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            feature_type (str): Type of asymmetry feature ('dasm', 'rasm', 'dcau')
            
        Returns:
            np.ndarray: Asymmetry features
        """
        # First compute DE features for asymmetry calculation
        de_features = self.compute_de_features(data)
        n_trials, n_features = de_features.shape
        n_bands = len(self.frequency_bands)
        n_channels = n_features // n_bands
        
        # Reshape to (n_trials, n_channels, n_bands)
        de_reshaped = de_features.reshape(n_trials, n_channels, n_bands)
        
        asymmetry_features = []
        
        # Define channel pairs for asymmetry computation
        # This is a simplified version - in practice, you'd use actual EEG channel locations
        left_channels = list(range(0, n_channels//2))
        right_channels = list(range(n_channels//2, n_channels))
        
        for trial in range(n_trials):
            trial_features = []
            
            for band_idx in range(n_bands):
                if feature_type == 'dasm':
                    # Differential Asymmetry: left - right
                    for left_ch, right_ch in zip(left_channels, right_channels):
                        if right_ch < n_channels:
                            asym_value = de_reshaped[trial, left_ch, band_idx] - de_reshaped[trial, right_ch, band_idx]
                            trial_features.append(asym_value)
                
                elif feature_type == 'rasm':
                    # Rational Asymmetry: (left - right) / (left + right)
                    for left_ch, right_ch in zip(left_channels, right_channels):
                        if right_ch < n_channels:
                            left_val = de_reshaped[trial, left_ch, band_idx]
                            right_val = de_reshaped[trial, right_ch, band_idx]
                            denominator = left_val + right_val + 1e-10
                            asym_value = (left_val - right_val) / denominator
                            trial_features.append(asym_value)
                
                elif feature_type == 'dcau':
                    # Caudality Asymmetry: frontal - posterior
                    frontal_channels = list(range(0, n_channels//3))
                    posterior_channels = list(range(2*n_channels//3, n_channels))
                    
                    frontal_mean = np.mean([de_reshaped[trial, ch, band_idx] for ch in frontal_channels])
                    posterior_mean = np.mean([de_reshaped[trial, ch, band_idx] for ch in posterior_channels])
                    
                    asym_value = frontal_mean - posterior_mean
                    trial_features.append(asym_value)
            
            asymmetry_features.append(trial_features)
        
        asymmetry_features = np.array(asymmetry_features)
        # logger.info(f"Computed {feature_type.upper()} features: shape {asymmetry_features.shape}")
        
        return asymmetry_features
    
    def extract_features(self, data: np.ndarray, feature_types: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract multiple types of features from EEG data.
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            feature_types (List[str]): List of feature types to extract
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with feature type names as keys and features as values
        """
        features = {}
        
        for feature_type in feature_types:
            try:
                if feature_type == 'psd':
                    features[feature_type] = self.compute_psd_features(data)
                elif feature_type == 'de':
                    features[feature_type] = self.compute_de_features(data)
                elif feature_type == 'dasm':
                    features[feature_type] = self.compute_asymmetry_features(data, 'dasm')
                elif feature_type == 'rasm':
                    features[feature_type] = self.compute_asymmetry_features(data, 'rasm')
                elif feature_type == 'dcau':
                    features[feature_type] = self.compute_asymmetry_features(data, 'dcau')
                elif feature_type == 'combined_asym':
                    # Combine all asymmetry features
                    dasm = self.compute_asymmetry_features(data, 'dasm')
                    rasm = self.compute_asymmetry_features(data, 'rasm')
                    dcau = self.compute_asymmetry_features(data, 'dcau')
                    features[feature_type] = np.concatenate([dasm, rasm, dcau], axis=1)
                else:
                    logger.warning(f"Unknown feature type: {feature_type}")
                    
            except Exception as e:
                logger.error(f"Error extracting {feature_type} features: {e}")
        
        return features
    
    def extract_band_specific_features(self, data: np.ndarray, band_name: str, feature_type: str = 'de') -> np.ndarray:
        """
        Extract features from a specific frequency band.
        
        Args:
            data (np.ndarray): EEG data with shape (n_trials, n_channels, n_timepoints)
            band_name (str): Name of frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
            feature_type (str): Type of feature to extract ('de', 'psd')
            
        Returns:
            np.ndarray: Band-specific features
        """
        if band_name not in self.frequency_bands:
            raise ValueError(f"Unknown frequency band: {band_name}")
        
        # Filter data to specific band
        band_data = self.extract_frequency_bands(data)
        band_signal = band_data[band_name]
        
        # Extract features from band-specific data
        if feature_type == 'de':
            # For single band DE, we compute DE directly on the filtered signal
            n_trials, n_channels, n_timepoints = band_signal.shape
            features = []
            
            for trial in range(n_trials):
                trial_features = []
                for ch in range(n_channels):
                    signal_data = band_signal[trial, ch, :] - np.mean(band_signal[trial, ch, :])
                    variance = np.var(signal_data)
                    de_value = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
                    trial_features.append(de_value)
                features.append(trial_features)
            
            features = np.array(features)
            
        elif feature_type == 'psd':
            features = self.compute_psd_features(band_signal)
        else:
            raise ValueError(f"Unsupported feature type for band-specific extraction: {feature_type}")
        
        # logger.info(f"Extracted {feature_type} features from {band_name} band: shape {features.shape}")
        return features