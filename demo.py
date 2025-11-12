#!/usr/bin/env python3
"""
SEED EEG Emotion Recognition - Demo Script

This script demonstrates the key functionality and generates sample results
for all four research objectives.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import SEEDDataLoader
from feature_extraction import EEGFeatureExtractor
from classifiers import EEGClassifier
from visualization import SEEDVisualizer

# Demo configuration
DEMO_CONFIG = {
    'data': {
        'input_dir': 'input/SEED_EEG',
        'output_dir': 'output',
        'results_dir': 'results',
        'sampling_rate': 200,
        'n_channels': 62,
        'n_subjects': 3,
        'frequency_bands': {
            'delta': [1, 3],
            'theta': [4, 7],
            'alpha': [8, 13],
            'beta': [14, 30],
            'gamma': [31, 50]
        }
    },
    'experiment': {
        'train_sessions': 3,
        'test_sessions': 2,
        'n_repeats': 3,
        'emotion_labels': {'sad': 0, 'neutral': 1, 'happy': 2},
        'random_seed': 42
    },
    'models': {
        'classifiers': ['svm', 'logistic', 'knn'],
        'svm': {'kernel': 'linear', 'C': 1.0, 'random_state': 42},
        'logistic': {'penalty': 'l2', 'C': 1.0, 'max_iter': 1000, 'random_state': 42},
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        'dbn': {'hidden_layers': [50, 25], 'learning_rate': 0.01, 'n_epochs': 20, 'batch_size': 32, 'random_state': 42}
    },
    'features': {
        'types': ['psd', 'de', 'dasm', 'rasm'],
        'window_length': 1.0,
        'overlap': 0.5
    },
    'visualization': {
        'figure_size': [12, 8],
        'dpi': 300,
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
        'file': 'logs/demo.log'
    }
}

def demo_frequency_analysis():
    """Demonstrate frequency band analysis."""
    print("🔬 Running Frequency Band Analysis Demo...")
    
    data_loader = SEEDDataLoader(DEMO_CONFIG['data']['input_dir'], DEMO_CONFIG)
    feature_extractor = EEGFeatureExtractor(DEMO_CONFIG)
    
    # Generate results for each frequency band
    results = {}
    bands = list(DEMO_CONFIG['data']['frequency_bands'].keys()) + ['all_5_bands']
    
    for band in bands:
        print(f"  Testing {band}...")
        accuracies = []
        
        for repeat in range(DEMO_CONFIG['experiment']['n_repeats']):
            # Generate synthetic data
            all_X, all_y = [], []
            
            for subject in range(1, DEMO_CONFIG['data']['n_subjects'] + 1):
                data, labels = data_loader.generate_synthetic_data(subject, 1)
                
                if band == 'all_5_bands':
                    features = feature_extractor.extract_features(data, ['de'])['de']
                else:
                    features = feature_extractor.extract_band_specific_features(data, band, 'de')
                
                all_X.extend(features)
                all_y.extend(labels)
            
            # Train and test classifier
            X = np.array(all_X)
            y = np.array(all_y)
            
            # Simple train/test split
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            classifier = EEGClassifier('svm', DEMO_CONFIG)
            classifier.train(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)
        
        results[band] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'all_accuracies': accuracies
        }
        
        print(f"    {band}: {results[band]['mean_accuracy']:.3f} ± {results[band]['std_accuracy']:.3f}")
    
    return results

def demo_channel_analysis():
    """Demonstrate channel montage analysis."""
    print("📡 Running Channel Montage Analysis Demo...")
    
    data_loader = SEEDDataLoader(DEMO_CONFIG['data']['input_dir'], DEMO_CONFIG)
    feature_extractor = EEGFeatureExtractor(DEMO_CONFIG)
    
    results = {}
    channel_counts = [4, 6, 9, 12, 62]
    
    for n_channels in channel_counts:
        print(f"  Testing {n_channels} channels...")
        accuracies = []
        
        for repeat in range(DEMO_CONFIG['experiment']['n_repeats']):
            all_X, all_y = [], []
            
            for subject in range(1, DEMO_CONFIG['data']['n_subjects'] + 1):
                data, labels = data_loader.generate_synthetic_data(subject, 1)
                
                # Reduce channels if needed
                if n_channels < 62:
                    data, _ = data_loader.get_channel_subset(data, n_channels)
                
                features = feature_extractor.extract_features(data, ['de'])['de']
                all_X.extend(features)
                all_y.extend(labels)
            
            X = np.array(all_X)
            y = np.array(all_y)
            
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            classifier = EEGClassifier('svm', DEMO_CONFIG)
            classifier.train(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)
        
        results[f'{n_channels}_channels'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'all_accuracies': accuracies
        }
        
        print(f"    {n_channels} channels: {results[f'{n_channels}_channels']['mean_accuracy']:.3f} ± {results[f'{n_channels}_channels']['std_accuracy']:.3f}")
    
    return results

def demo_classifier_comparison():
    """Demonstrate classifier comparison."""
    print("🤖 Running Classifier Comparison Demo...")
    
    data_loader = SEEDDataLoader(DEMO_CONFIG['data']['input_dir'], DEMO_CONFIG)
    feature_extractor = EEGFeatureExtractor(DEMO_CONFIG)
    
    results = {}
    classifiers = DEMO_CONFIG['models']['classifiers']
    
    for clf_type in classifiers:
        print(f"  Testing {clf_type}...")
        accuracies = []
        
        for repeat in range(DEMO_CONFIG['experiment']['n_repeats']):
            all_X, all_y = [], []
            
            for subject in range(1, DEMO_CONFIG['data']['n_subjects'] + 1):
                data, labels = data_loader.generate_synthetic_data(subject, 1)
                features = feature_extractor.extract_features(data, ['de'])['de']
                all_X.extend(features)
                all_y.extend(labels)
            
            X = np.array(all_X)
            y = np.array(all_y)
            
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            classifier = EEGClassifier(clf_type, DEMO_CONFIG)
            classifier.train(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)
        
        results[clf_type] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'all_accuracies': accuracies
        }
        
        print(f"    {clf_type}: {results[clf_type]['mean_accuracy']:.3f} ± {results[clf_type]['std_accuracy']:.3f}")
    
    return results

def demo_feature_comparison():
    """Demonstrate feature set comparison."""
    print("🔍 Running Feature Set Comparison Demo...")
    
    data_loader = SEEDDataLoader(DEMO_CONFIG['data']['input_dir'], DEMO_CONFIG)
    feature_extractor = EEGFeatureExtractor(DEMO_CONFIG)
    
    results = {}
    feature_types = DEMO_CONFIG['features']['types']
    
    for feat_type in feature_types:
        print(f"  Testing {feat_type} features...")
        accuracies = []
        
        for repeat in range(DEMO_CONFIG['experiment']['n_repeats']):
            all_X, all_y = [], []
            
            for subject in range(1, DEMO_CONFIG['data']['n_subjects'] + 1):
                data, labels = data_loader.generate_synthetic_data(subject, 1)
                features = feature_extractor.extract_features(data, [feat_type])[feat_type]
                all_X.extend(features)
                all_y.extend(labels)
            
            X = np.array(all_X)
            y = np.array(all_y)
            
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            classifier = EEGClassifier('svm', DEMO_CONFIG)
            classifier.train(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            accuracies.append(accuracy)
        
        results[feat_type] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'all_accuracies': accuracies
        }
        
        print(f"    {feat_type}: {results[feat_type]['mean_accuracy']:.3f} ± {results[feat_type]['std_accuracy']:.3f}")
    
    return results

def save_results(results, filename):
    """Save results to JSON file."""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(output_dir / filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"  💾 Results saved to output/{filename}")

def create_visualizations(all_results):
    """Create visualizations for all results."""
    print("📊 Creating Visualizations...")
    
    visualizer = SEEDVisualizer(DEMO_CONFIG, 'output')
    
    # Create individual plots
    visualizer.plot_frequency_band_analysis(all_results['frequency_analysis'])
    visualizer.plot_channel_analysis(all_results['channel_analysis'])
    visualizer.plot_classifier_comparison(all_results['classifier_comparison'])
    visualizer.plot_feature_comparison(all_results['feature_comparison'])
    
    # Create comprehensive summary
    visualizer.plot_comprehensive_summary(all_results)
    
    print("  📈 All visualizations created!")

def main():
    """Run the complete demo."""
    print("=" * 80)
    print("🧠 SEED EEG Emotion Recognition - Research Demo")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Create output directories
    Path('output').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Run all demonstrations
    freq_results = demo_frequency_analysis()
    save_results(freq_results, 'frequency_analysis_demo.json')
    
    channel_results = demo_channel_analysis()
    save_results(channel_results, 'channel_analysis_demo.json')
    
    classifier_results = demo_classifier_comparison()
    save_results(classifier_results, 'classifier_comparison_demo.json')
    
    feature_results = demo_feature_comparison()
    save_results(feature_results, 'feature_comparison_demo.json')
    
    # Compile all results
    all_results = {
        'frequency_analysis': freq_results,
        'channel_analysis': channel_results,
        'classifier_comparison': classifier_results,
        'feature_comparison': feature_results,
        'experiment_info': {
            'total_runtime': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Demo results using synthetic data'
        }
    }
    
    save_results(all_results, 'complete_demo_results.json')
    
    # Create visualizations
    create_visualizations(all_results)
    
    # Print summary
    total_time = time.time() - start_time
    print()
    print("=" * 80)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"⏱️  Total Runtime: {total_time:.1f} seconds")
    print(f"📁 Results saved to: output/")
    print()
    print("🔬 Key Findings (Demo with Synthetic Data):")
    
    # Find best results
    best_freq = max(freq_results.items(), key=lambda x: x[1]['mean_accuracy'])
    best_channels = max(channel_results.items(), key=lambda x: x[1]['mean_accuracy'])
    best_classifier = max(classifier_results.items(), key=lambda x: x[1]['mean_accuracy'])
    best_feature = max(feature_results.items(), key=lambda x: x[1]['mean_accuracy'])
    
    print(f"  🎯 Best frequency band: {best_freq[0]} ({best_freq[1]['mean_accuracy']:.3f})")
    print(f"  📡 Optimal channels: {best_channels[0]} ({best_channels[1]['mean_accuracy']:.3f})")
    print(f"  🤖 Best classifier: {best_classifier[0]} ({best_classifier[1]['mean_accuracy']:.3f})")
    print(f"  🔍 Best features: {best_feature[0]} ({best_feature[1]['mean_accuracy']:.3f})")
    print()
    print("📊 Check the output/ directory for detailed visualizations!")
    print("=" * 80)

if __name__ == "__main__":
    main()