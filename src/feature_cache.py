"""
Feature Cache Module

This module provides caching functionality for extracted EEG features to avoid recomputation.
"""

import os
import pickle
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Configure logging to be less verbose
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FeatureCache:
    """
    Caches extracted EEG features to disk for reuse.
    """
    
    def __init__(self, cache_dir: str = "data/feature_cache"):
        """
        Initialize feature cache.
        
        Args:
            cache_dir (str): Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized feature cache at {self.cache_dir}")
    
    def _generate_cache_key(self, data: np.ndarray, config: Dict[str, Any]) -> str:
        """
        Generate unique cache key based on data and configuration.
        
        Args:
            data (np.ndarray): Input EEG data
            config (Dict[str, Any]): Feature extraction configuration
            
        Returns:
            str: Unique cache key
        """
        # Create hash from data shape, config, and data sample
        data_info = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'sample': data.flat[:min(100, data.size)].tolist()  # Sample first 100 elements
        }
        
        # Combine data info and config
        cache_input = {
            'data_info': data_info,
            'config': config
        }
        
        # Generate hash
        cache_str = str(sorted(cache_input.items()))
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_key
    
    def get_features(self, data: np.ndarray, config: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve cached features if available.
        
        Args:
            data (np.ndarray): Input EEG data
            config (Dict[str, Any]): Feature extraction configuration
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Cached features or None if not found
        """
        cache_key = self._generate_cache_key(data, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_features = pickle.load(f)
                logger.debug(f"Loaded cached features from {cache_file}")
                return cached_features
            except Exception as e:
                logger.warning(f"Failed to load cached features: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def save_features(self, data: np.ndarray, config: Dict[str, Any], 
                     features: Dict[str, np.ndarray]) -> None:
        """
        Save extracted features to cache.
        
        Args:
            data (np.ndarray): Input EEG data
            config (Dict[str, Any]): Feature extraction configuration
            features (Dict[str, np.ndarray]): Extracted features to cache
        """
        cache_key = self._generate_cache_key(data, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            logger.debug(f"Saved features to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save features to cache: {e}")
    
    def clear_cache(self) -> None:
        """
        Clear all cached features.
        """
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared feature cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached features.
        
        Returns:
            Dict[str, Any]: Cache information
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_cached_features': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


class OptimizedFeatureExtractor:
    """
    Feature extractor with caching and optimization.
    """
    
    def __init__(self, cache_dir: str = "data/feature_cache"):
        """
        Initialize optimized feature extractor.
        
        Args:
            cache_dir (str): Directory for feature cache
        """
        self.cache = FeatureCache(cache_dir)
        
        # Import the original feature extractor
        from .feature_extraction import EEGFeatureExtractor
        self.extractor = EEGFeatureExtractor()
    
    def extract_features(self, data: np.ndarray, feature_types: list, 
                        config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract features with caching.
        
        Args:
            data (np.ndarray): EEG data (channels, samples)
            feature_types (list): List of feature types to extract
            config (Dict[str, Any]): Configuration parameters
            
        Returns:
            Dict[str, np.ndarray]: Extracted features
        """
        # Create cache key config
        cache_config = {
            'feature_types': sorted(feature_types),
            'sampling_rate': config.get('sampling_rate', 200),
            'freq_bands': config.get('freq_bands', {}),
            'window_size': config.get('window_size', 1.0)
        }
        
        # Try to get cached features
        cached_features = self.cache.get_features(data, cache_config)
        if cached_features is not None:
            # Filter to requested feature types
            return {ft: cached_features[ft] for ft in feature_types if ft in cached_features}
        
        # Extract features if not cached
        logger.debug("Extracting features (not cached)")
        features = {}
        
        for feature_type in feature_types:
            if feature_type == 'PSD':
                features[feature_type] = self.extractor.extract_psd_features(data, config)
            elif feature_type == 'DE':
                features[feature_type] = self.extractor.extract_de_features(data, config)
            elif feature_type == 'DASM':
                features[feature_type] = self.extractor.extract_dasm_features(data, config)
            elif feature_type == 'RASM':
                features[feature_type] = self.extractor.extract_rasm_features(data, config)
            elif feature_type == 'DCAU':
                features[feature_type] = self.extractor.extract_dcau_features(data, config)
        
        # Cache the extracted features
        self.cache.save_features(data, cache_config, features)
        
        return features
    
    def batch_extract_features(self, data_list: list, feature_types: list, 
                             config: Dict[str, Any]) -> list:
        """
        Extract features for multiple data samples efficiently.
        
        Args:
            data_list (list): List of EEG data arrays
            feature_types (list): List of feature types to extract
            config (Dict[str, Any]): Configuration parameters
            
        Returns:
            list: List of extracted feature dictionaries
        """
        results = []
        
        for i, data in enumerate(data_list):
            if i % 10 == 0:  # Progress logging every 10 samples
                logger.debug(f"Processing sample {i+1}/{len(data_list)}")
            
            features = self.extract_features(data, feature_types, config)
            results.append(features)
        
        return results