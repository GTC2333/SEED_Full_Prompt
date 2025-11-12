#!/usr/bin/env python3
"""
Quick Test Script for SEED EEG Emotion Recognition

This script performs a quick test of all major components to ensure
the system is working correctly before running full experiments.
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import SEEDDataLoader
from feature_extraction import EEGFeatureExtractor
from classifiers import EEGClassifier
from visualization import SEEDVisualizer

# Simple test configuration
TEST_CONFIG = {
    'data': {
        'input_dir': 'input/SEED_EEG',
        'output_dir': 'output',
        'results_dir': 'results',
        'sampling_rate': 200,
        'n_channels': 62,
        'n_subjects': 2,  # Reduced for testing
        'frequency_bands': {
            'alpha': [8, 13],
            'beta': [14, 30]
        }
    },
    'experiment': {
        'train_sessions': 3,
        'test_sessions': 2,
        'n_repeats': 2,
        'emotion_labels': {'sad': 0, 'neutral': 1, 'happy': 2},
        'random_seed': 42
    },
    'models': {
        'classifiers': ['svm', 'logistic'],
        'svm': {'kernel': 'linear', 'C': 1.0, 'random_state': 42},
        'logistic': {'penalty': 'l2', 'C': 1.0, 'max_iter': 1000, 'random_state': 42},
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        'dbn': {'hidden_layers': [50, 25], 'learning_rate': 0.01, 'n_epochs': 10, 'batch_size': 32, 'random_state': 42}
    },
    'features': {
        'types': ['de', 'psd'],
        'window_length': 1.0,
        'overlap': 0.5
    },
    'visualization': {
        'figure_size': [10, 6],
        'dpi': 150,
        'colors': {'sad': '#FF6B6B', 'neutral': '#4ECDC4', 'happy': '#45B7D1'},
        'save_formats': ['png']
    },
    'computation': {
        'n_jobs': 1,
        'batch_processing': True,
        'batch_size': 100,
        'use_gpu': False,
        'device': 'cpu'
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/test.log'
    }
}

def test_data_loader():
    """Test the data loader component."""
    print("Testing Data Loader...")
    
    data_loader = SEEDDataLoader(TEST_CONFIG['data']['input_dir'], TEST_CONFIG)
    
    # Test synthetic data generation
    data, labels = data_loader.generate_synthetic_data(1, 1)
    print(f"  ✓ Generated synthetic data: shape {data.shape}, labels {np.bincount(labels)}")
    
    # Test channel subset selection
    reduced_data, selected_channels = data_loader.get_channel_subset(data, 4)
    print(f"  ✓ Channel subset: {reduced_data.shape}, channels {selected_channels}")
    
    # Test loading subject session
    data, labels = data_loader.load_subject_session(1, 1)
    print(f"  ✓ Loaded subject session: shape {data.shape}")
    
    return data_loader

def test_feature_extraction(data_loader):
    """Test the feature extraction component."""
    print("Testing Feature Extraction...")
    
    feature_extractor = EEGFeatureExtractor(TEST_CONFIG)
    
    # Get test data
    data, labels = data_loader.generate_synthetic_data(1, 1)
    
    # Test DE features
    de_features = feature_extractor.compute_de_features(data)
    print(f"  ✓ DE features: shape {de_features.shape}")
    
    # Test PSD features
    psd_features = feature_extractor.compute_psd_features(data)
    print(f"  ✓ PSD features: shape {psd_features.shape}")
    
    # Test multiple feature extraction
    features = feature_extractor.extract_features(data, ['de', 'psd'])
    print(f"  ✓ Multiple features: {list(features.keys())}")
    
    return feature_extractor, features

def test_classifiers(features, labels):
    """Test the classifier components."""
    print("Testing Classifiers...")
    
    # Get feature data
    X = features['de']
    y = labels
    
    # Split data
    split_idx = len(X) // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test SVM classifier
    svm_clf = EEGClassifier('svm', TEST_CONFIG)
    svm_clf.train(X_train, y_train)
    svm_predictions = svm_clf.predict(X_test)
    svm_accuracy = np.mean(svm_predictions == y_test)
    print(f"  ✓ SVM classifier: accuracy {svm_accuracy:.3f}")
    
    # Test Logistic Regression classifier
    lr_clf = EEGClassifier('logistic', TEST_CONFIG)
    lr_clf.train(X_train, y_train)
    lr_predictions = lr_clf.predict(X_test)
    lr_accuracy = np.mean(lr_predictions == y_test)
    print(f"  ✓ Logistic Regression: accuracy {lr_accuracy:.3f}")
    
    return {'svm': svm_accuracy, 'logistic': lr_accuracy}

def test_visualization(results):
    """Test the visualization component."""
    print("Testing Visualization...")
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    visualizer = SEEDVisualizer(TEST_CONFIG, 'output')
    
    # Create mock results for testing
    mock_results = {
        'alpha': {'mean_accuracy': 0.65, 'std_accuracy': 0.05, 'all_accuracies': [0.6, 0.7]},
        'beta': {'mean_accuracy': 0.70, 'std_accuracy': 0.04, 'all_accuracies': [0.66, 0.74]}
    }
    
    # Test frequency band plot
    visualizer.plot_frequency_band_analysis(mock_results)
    print("  ✓ Created frequency band analysis plot")
    
    # Test classifier comparison plot
    classifier_results = {
        'svm': {'mean_accuracy': results['svm'], 'std_accuracy': 0.02, 'all_accuracies': [results['svm']]},
        'logistic': {'mean_accuracy': results['logistic'], 'std_accuracy': 0.03, 'all_accuracies': [results['logistic']]}
    }
    visualizer.plot_classifier_comparison(classifier_results)
    print("  ✓ Created classifier comparison plot")

def main():
    """Run all tests."""
    print("=" * 60)
    print("SEED EEG Emotion Recognition - Quick Test")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # Test components
        data_loader = test_data_loader()
        feature_extractor, features = test_feature_extraction(data_loader)
        
        # Get labels for testing
        _, labels = data_loader.generate_synthetic_data(1, 1)
        
        classifier_results = test_classifiers(features, labels)
        test_visualization(classifier_results)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - System is ready for full experiments!")
        print("=" * 60)
        
        print("\nTo run full experiments:")
        print("  python main.py --quick-test    # Quick test with reduced parameters")
        print("  python main.py                 # Full experiment suite")
        print("  python main.py --objective 1   # Run specific research objective")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()