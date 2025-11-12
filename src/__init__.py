"""
SEED EEG Emotion Recognition Package

This package implements a comprehensive research framework for EEG-based emotion recognition
using the SEED dataset. It includes data loading, feature extraction, classification,
and visualization components.
"""

__version__ = "1.0.0"
__author__ = "SEED Research Team"
__email__ = "research@seed-eeg.org"

from .data_loader import SEEDDataLoader
from .feature_extraction import EEGFeatureExtractor
from .classifiers import EEGClassifier, ClassifierComparison, DeepBeliefNetwork
from .experiments import SEEDExperiments
from .visualization import SEEDVisualizer

__all__ = [
    'SEEDDataLoader',
    'EEGFeatureExtractor', 
    'EEGClassifier',
    'ClassifierComparison',
    'DeepBeliefNetwork',
    'SEEDExperiments',
    'SEEDVisualizer'
]