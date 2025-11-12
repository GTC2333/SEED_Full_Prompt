"""
SEED Dataset Data Loader Module

This module handles loading and preprocessing of SEED EEG emotion recognition dataset.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SEEDDataLoader:
    """
    Data loader for SEED EEG emotion recognition dataset.
    
    Handles loading .mat files, extracting EEG data and labels,
    and organizing data for machine learning experiments.
    """
    
    def __init__(self, data_dir: str, config: Dict):
        """
        Initialize SEED data loader.
        
        Args:
            data_dir (str): Path to SEED dataset directory
            config (Dict): Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.channels = self._load_channel_order()
        self.stimulation_info = self._load_stimulation_info()
        
        # EEG parameters
        self.n_channels = config['data']['n_channels']
        self.sampling_rate = config['data']['sampling_rate']
        self.n_subjects = config['data']['n_subjects']
        
        logger.info(f"Initialized SEED data loader for {self.n_subjects} subjects")
    
    def _load_channel_order(self) -> List[str]:
        """
        Load EEG channel order from CSV file.
        
        Returns:
            List[str]: List of channel names in order
        """
        try:
            channel_file = self.data_dir / "channel-order.csv"
            if channel_file.exists():
                df = pd.read_csv(channel_file)
                return df['Channel'].tolist()
            else:
                # Default 62-channel montage if file not found
                logger.warning("Channel order file not found, using default montage")
                return [f"CH{i+1}" for i in range(62)]
        except Exception as e:
            logger.error(f"Error loading channel order: {e}")
            return [f"CH{i+1}" for i in range(62)]
    
    def _load_stimulation_info(self) -> pd.DataFrame:
        """
        Load stimulation information from CSV file.
        
        Returns:
            pd.DataFrame: Stimulation information with labels
        """
        try:
            stim_file = self.data_dir / "SEED_stimulation.csv"
            if stim_file.exists():
                return pd.read_csv(stim_file)
            else:
                logger.warning("Stimulation file not found, creating default")
                # Create default stimulation info
                return pd.DataFrame({
                    'Name of the clip': ['Default'] * 15,
                    'Label': [0, 1, 2] * 5,  # Balanced labels
                    'Start time point': ['0:00:00'] * 15,
                    'End time point': ['0:04:00'] * 15
                })
        except Exception as e:
            logger.error(f"Error loading stimulation info: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_data(self, subject_id: int, session_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EEG data for testing when real data is not available.
        
        Args:
            subject_id (int): Subject identifier (1-15)
            session_id (int): Session identifier (1-15)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (data, labels)
                - data: Shape (n_trials, n_channels, n_timepoints)
                - labels: Shape (n_trials,)
        """
        np.random.seed(42 + subject_id * 100 + session_id)
        
        # Generate 15 trials (5 per emotion class)
        n_trials = 15
        n_timepoints = 800  # 4 seconds at 200 Hz
        
        # Create synthetic EEG data with realistic characteristics
        data = np.zeros((n_trials, self.n_channels, n_timepoints))
        labels = np.array([0, 1, 2] * 5)  # Balanced labels
        
        for trial in range(n_trials):
            emotion = labels[trial]
            
            # Generate base EEG signal with different characteristics per emotion
            for ch in range(self.n_channels):
                # Base signal with 1/f noise
                freqs = np.fft.fftfreq(n_timepoints, 1/self.sampling_rate)
                power_spectrum = 1 / (np.abs(freqs) + 1)
                power_spectrum[0] = 0  # Remove DC component
                
                # Add emotion-specific frequency content
                if emotion == 0:  # Sad - more low frequency activity
                    power_spectrum[np.abs(freqs) < 8] *= 2.0
                elif emotion == 1:  # Neutral - balanced
                    pass
                elif emotion == 2:  # Happy - more high frequency activity
                    power_spectrum[np.abs(freqs) > 13] *= 1.5
                
                # Generate random phases
                phases = np.random.uniform(0, 2*np.pi, len(freqs))
                complex_spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
                
                # Convert to time domain
                signal = np.fft.ifft(complex_spectrum).real
                
                # Add some spatial correlation between channels
                if ch > 0:
                    signal += 0.3 * data[trial, ch-1, :] + 0.1 * np.random.randn(n_timepoints)
                
                data[trial, ch, :] = signal
        
        # Normalize data
        data = (data - np.mean(data, axis=2, keepdims=True)) / np.std(data, axis=2, keepdims=True)
        
        logger.info(f"Generated synthetic data for subject {subject_id}, session {session_id}: "
                   f"shape {data.shape}, labels {np.bincount(labels)}")
        
        return data, labels
    
    def load_subject_session(self, subject_id: int, session_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG data for a specific subject and session.
        
        Args:
            subject_id (int): Subject identifier (1-15)
            session_id (int): Session identifier (1-15)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (data, labels)
                - data: Shape (n_trials, n_channels, n_timepoints)
                - labels: Shape (n_trials,)
        """
        # Try to load real .mat file
        mat_file = self.data_dir / "Preprocessed_EEG" / f"{subject_id}_{session_id:08d}.mat"
        
        if mat_file.exists():
            try:
                mat_data = sio.loadmat(str(mat_file))
                # Extract data and labels from .mat file
                # This would need to be adapted based on actual SEED .mat file structure
                data = mat_data.get('data', None)
                labels = mat_data.get('labels', None)
                
                if data is not None and labels is not None:
                    logger.info(f"Loaded real data for subject {subject_id}, session {session_id}")
                    return data, labels
            except Exception as e:
                logger.warning(f"Error loading .mat file {mat_file}: {e}")
        
        # Fall back to synthetic data
        logger.info(f"Using synthetic data for subject {subject_id}, session {session_id}")
        return self.generate_synthetic_data(subject_id, session_id)
    
    def load_all_data(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Load all available EEG data for all subjects and sessions.
        
        Returns:
            Dict: Nested dictionary with structure:
                {subject_id: {session_id: (data, labels)}}
        """
        all_data = {}
        
        for subject_id in range(1, self.n_subjects + 1):
            all_data[subject_id] = {}
            
            for session_id in range(1, 16):  # 15 sessions per subject
                try:
                    data, labels = self.load_subject_session(subject_id, session_id)
                    all_data[subject_id][session_id] = (data, labels)
                except Exception as e:
                    logger.error(f"Failed to load subject {subject_id}, session {session_id}: {e}")
        
        logger.info(f"Loaded data for {len(all_data)} subjects")
        return all_data
    
    def get_channel_subset(self, data: np.ndarray, n_channels: int) -> Tuple[np.ndarray, List[str]]:
        """
        Select a subset of channels for reduced montage experiments.
        
        Args:
            data (np.ndarray): Full EEG data with shape (..., n_channels, ...)
            n_channels (int): Number of channels to select
            
        Returns:
            Tuple[np.ndarray, List[str]]: (reduced_data, selected_channels)
        """
        if n_channels >= self.n_channels:
            return data, self.channels
        
        # Select channels based on standard EEG montages
        if n_channels == 4:
            # Basic frontal-central-parietal-occipital
            selected_indices = [9, 18, 27, 45]  # FZ, FCZ, CZ, PZ
            selected_channels = ['FZ', 'FCZ', 'CZ', 'PZ']
        elif n_channels == 6:
            # Add temporal channels
            selected_indices = [9, 18, 27, 45, 23, 31]  # FZ, FCZ, CZ, PZ, T7, T8
            selected_channels = ['FZ', 'FCZ', 'CZ', 'PZ', 'T7', 'T8']
        elif n_channels == 9:
            # 3x3 grid
            selected_indices = [8, 9, 10, 26, 27, 28, 44, 45, 46]  # F1, FZ, F2, C1, CZ, C2, P1, PZ, P2
            selected_channels = ['F1', 'FZ', 'F2', 'C1', 'CZ', 'C2', 'P1', 'PZ', 'P2']
        elif n_channels == 12:
            # Extended montage
            selected_indices = [8, 9, 10, 18, 26, 27, 28, 36, 44, 45, 46, 53]
            selected_channels = ['F1', 'FZ', 'F2', 'FCZ', 'C1', 'CZ', 'C2', 'CPZ', 'P1', 'PZ', 'P2', 'POZ']
        else:
            # Random selection for other numbers
            selected_indices = np.random.choice(self.n_channels, n_channels, replace=False)
            selected_channels = [self.channels[i] for i in selected_indices]
        
        # Extract selected channels
        reduced_data = data[..., selected_indices, :]
        
        logger.info(f"Selected {n_channels} channels: {selected_channels}")
        return reduced_data, selected_channels
    
    def get_data_info(self) -> Dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dict: Dataset information
        """
        return {
            'n_subjects': self.n_subjects,
            'n_channels': self.n_channels,
            'sampling_rate': self.sampling_rate,
            'channels': self.channels,
            'emotion_labels': self.config['experiment']['emotion_labels']
        }