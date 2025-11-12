"""
Experiments Module for SEED EEG Emotion Recognition

This module implements the four main research objectives:
1. Frequency band analysis
2. Channel montage optimization
3. Classifier comparison
4. Feature set comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import logging
import json
from pathlib import Path
import time

from data_loader import SEEDDataLoader
from feature_extraction import EEGFeatureExtractor
from classifiers import EEGClassifier, ClassifierComparison

logger = logging.getLogger(__name__)

class SEEDExperiments:
    """
    Main experiment runner for SEED EEG emotion recognition research.
    
    Implements all four research objectives with proper experimental design
    and statistical analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize experiment runner.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.data_loader = SEEDDataLoader(config['data']['input_dir'], config)
        self.feature_extractor = EEGFeatureExtractor(config)
        self.results = {}
        
        # Create output directories
        self.output_dir = Path(config['data']['output_dir'])
        self.results_dir = Path(config['data']['results_dir'])
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Initialized SEED experiments")
    
    def run_frequency_band_analysis(self) -> Dict[str, Any]:
        """
        Research Objective 1: Identify which EEG frequency bands carry 
        the most discriminative information for emotion recognition.
        
        Returns:
            Dict[str, Any]: Results of frequency band analysis
        """
        logger.info("Starting frequency band analysis...")
        
        # Load all data
        all_data = self.data_loader.load_all_data()
        
        # Test each frequency band individually and in combination
        band_names = list(self.config['data']['frequency_bands'].keys())
        test_conditions = band_names + ['all_5_bands']
        
        results = {}
        
        for condition in test_conditions:
            logger.info(f"Testing frequency condition: {condition}")
            condition_results = []
            
            # Run multiple cross-validation repeats
            for repeat in range(self.config['experiment']['n_repeats']):
                logger.info(f"Repeat {repeat + 1}/{self.config['experiment']['n_repeats']}")
                
                # Collect data from all subjects
                X_all, y_all = [], []
                
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    
                    # Split sessions into train/test
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    
                    train_sessions = session_ids[:self.config['experiment']['train_sessions']]
                    test_sessions = session_ids[self.config['experiment']['train_sessions']:]
                    
                    # Collect training data
                    for session_id in train_sessions:
                        data, labels = subject_data[session_id]
                        
                        if condition == 'all_5_bands':
                            # Use all frequency bands
                            features = self.feature_extractor.extract_features(data, ['de'])['de']
                        else:
                            # Use specific frequency band
                            features = self.feature_extractor.extract_band_specific_features(
                                data, condition, 'de'
                            )
                        
                        X_all.extend(features)
                        y_all.extend(labels)
                
                # Convert to numpy arrays
                X_all = np.array(X_all)
                y_all = np.array(y_all)
                
                # Train classifier (using SVM as baseline)
                classifier = EEGClassifier('svm', self.config)
                classifier.train(X_all, y_all)
                
                # Test on remaining sessions
                test_accuracies = []
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    test_sessions = session_ids[self.config['experiment']['train_sessions']:]
                    
                    for session_id in test_sessions:
                        data, labels = subject_data[session_id]
                        
                        if condition == 'all_5_bands':
                            features = self.feature_extractor.extract_features(data, ['de'])['de']
                        else:
                            features = self.feature_extractor.extract_band_specific_features(
                                data, condition, 'de'
                            )
                        
                        predictions = classifier.predict(features)
                        accuracy = np.mean(predictions == labels)
                        test_accuracies.append(accuracy)
                
                condition_results.append(np.mean(test_accuracies))
            
            results[condition] = {
                'mean_accuracy': np.mean(condition_results),
                'std_accuracy': np.std(condition_results),
                'all_accuracies': condition_results
            }
            
            logger.info(f"{condition}: {results[condition]['mean_accuracy']:.4f} ± {results[condition]['std_accuracy']:.4f}")
        
        # Save results
        self.results['frequency_analysis'] = results
        self._save_results('frequency_analysis', results)
        
        logger.info("Completed frequency band analysis")
        return results
    
    def run_channel_analysis(self) -> Dict[str, Any]:
        """
        Research Objective 2: Identify minimal electrode montage that retains
        emotion-discrimination capability.
        
        Returns:
            Dict[str, Any]: Results of channel analysis
        """
        logger.info("Starting channel montage analysis...")
        
        all_data = self.data_loader.load_all_data()
        channel_counts = [4, 6, 9, 12, 62]  # Different montage sizes
        
        results = {}
        
        for n_channels in channel_counts:
            logger.info(f"Testing {n_channels}-channel montage")
            condition_results = []
            
            for repeat in range(self.config['experiment']['n_repeats']):
                logger.info(f"Repeat {repeat + 1}/{self.config['experiment']['n_repeats']}")
                
                # Collect and process data
                X_all, y_all = [], []
                
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    
                    # Split sessions
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    
                    train_sessions = session_ids[:self.config['experiment']['train_sessions']]
                    
                    for session_id in train_sessions:
                        data, labels = subject_data[session_id]
                        
                        # Reduce channel count
                        if n_channels < 62:
                            data, _ = self.data_loader.get_channel_subset(data, n_channels)
                        
                        # Extract DE features
                        features = self.feature_extractor.extract_features(data, ['de'])['de']
                        
                        X_all.extend(features)
                        y_all.extend(labels)
                
                X_all = np.array(X_all)
                y_all = np.array(y_all)
                
                # Train classifier
                classifier = EEGClassifier('svm', self.config)
                classifier.train(X_all, y_all)
                
                # Test
                test_accuracies = []
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    test_sessions = session_ids[self.config['experiment']['train_sessions']:]
                    
                    for session_id in test_sessions:
                        data, labels = subject_data[session_id]
                        
                        if n_channels < 62:
                            data, _ = self.data_loader.get_channel_subset(data, n_channels)
                        
                        features = self.feature_extractor.extract_features(data, ['de'])['de']
                        predictions = classifier.predict(features)
                        accuracy = np.mean(predictions == labels)
                        test_accuracies.append(accuracy)
                
                condition_results.append(np.mean(test_accuracies))
            
            results[f'{n_channels}_channels'] = {
                'mean_accuracy': np.mean(condition_results),
                'std_accuracy': np.std(condition_results),
                'all_accuracies': condition_results
            }
            
            logger.info(f"{n_channels} channels: {results[f'{n_channels}_channels']['mean_accuracy']:.4f} ± {results[f'{n_channels}_channels']['std_accuracy']:.4f}")
        
        self.results['channel_analysis'] = results
        self._save_results('channel_analysis', results)
        
        logger.info("Completed channel montage analysis")
        return results
    
    def run_classifier_comparison(self) -> Dict[str, Any]:
        """
        Research Objective 3: Compare deep vs shallow methods for 
        EEG-based emotion recognition.
        
        Returns:
            Dict[str, Any]: Results of classifier comparison
        """
        logger.info("Starting classifier comparison...")
        
        all_data = self.data_loader.load_all_data()
        classifier_types = self.config['models']['classifiers']
        
        results = {}
        
        for classifier_type in classifier_types:
            logger.info(f"Testing {classifier_type} classifier")
            condition_results = []
            
            for repeat in range(self.config['experiment']['n_repeats']):
                logger.info(f"Repeat {repeat + 1}/{self.config['experiment']['n_repeats']}")
                
                # Collect data
                X_all, y_all = [], []
                
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    
                    train_sessions = session_ids[:self.config['experiment']['train_sessions']]
                    
                    for session_id in train_sessions:
                        data, labels = subject_data[session_id]
                        features = self.feature_extractor.extract_features(data, ['de'])['de']
                        
                        X_all.extend(features)
                        y_all.extend(labels)
                
                X_all = np.array(X_all)
                y_all = np.array(y_all)
                
                # Train classifier
                classifier = EEGClassifier(classifier_type, self.config)
                classifier.train(X_all, y_all)
                
                # Test
                test_accuracies = []
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    test_sessions = session_ids[self.config['experiment']['train_sessions']:]
                    
                    for session_id in test_sessions:
                        data, labels = subject_data[session_id]
                        features = self.feature_extractor.extract_features(data, ['de'])['de']
                        predictions = classifier.predict(features)
                        accuracy = np.mean(predictions == labels)
                        test_accuracies.append(accuracy)
                
                condition_results.append(np.mean(test_accuracies))
            
            results[classifier_type] = {
                'mean_accuracy': np.mean(condition_results),
                'std_accuracy': np.std(condition_results),
                'all_accuracies': condition_results
            }
            
            logger.info(f"{classifier_type}: {results[classifier_type]['mean_accuracy']:.4f} ± {results[classifier_type]['std_accuracy']:.4f}")
        
        self.results['classifier_comparison'] = results
        self._save_results('classifier_comparison', results)
        
        logger.info("Completed classifier comparison")
        return results
    
    def run_feature_comparison(self) -> Dict[str, Any]:
        """
        Research Objective 4: Establish which EEG feature set conveys 
        the most discriminative information.
        
        Returns:
            Dict[str, Any]: Results of feature comparison
        """
        logger.info("Starting feature set comparison...")
        
        all_data = self.data_loader.load_all_data()
        feature_types = self.config['features']['types']
        
        results = {}
        
        for feature_type in feature_types:
            logger.info(f"Testing {feature_type} features")
            condition_results = []
            
            for repeat in range(self.config['experiment']['n_repeats']):
                logger.info(f"Repeat {repeat + 1}/{self.config['experiment']['n_repeats']}")
                
                # Collect data
                X_all, y_all = [], []
                
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    
                    train_sessions = session_ids[:self.config['experiment']['train_sessions']]
                    
                    for session_id in train_sessions:
                        data, labels = subject_data[session_id]
                        features = self.feature_extractor.extract_features(data, [feature_type])[feature_type]
                        
                        X_all.extend(features)
                        y_all.extend(labels)
                
                X_all = np.array(X_all)
                y_all = np.array(y_all)
                
                # Train SVM classifier
                classifier = EEGClassifier('svm', self.config)
                classifier.train(X_all, y_all)
                
                # Test
                test_accuracies = []
                for subject_id in range(1, self.config['data']['n_subjects'] + 1):
                    subject_data = all_data[subject_id]
                    session_ids = list(subject_data.keys())
                    np.random.seed(self.config['experiment']['random_seed'] + repeat)
                    np.random.shuffle(session_ids)
                    test_sessions = session_ids[self.config['experiment']['train_sessions']:]
                    
                    for session_id in test_sessions:
                        data, labels = subject_data[session_id]
                        features = self.feature_extractor.extract_features(data, [feature_type])[feature_type]
                        predictions = classifier.predict(features)
                        accuracy = np.mean(predictions == labels)
                        test_accuracies.append(accuracy)
                
                condition_results.append(np.mean(test_accuracies))
            
            results[feature_type] = {
                'mean_accuracy': np.mean(condition_results),
                'std_accuracy': np.std(condition_results),
                'all_accuracies': condition_results
            }
            
            logger.info(f"{feature_type}: {results[feature_type]['mean_accuracy']:.4f} ± {results[feature_type]['std_accuracy']:.4f}")
        
        self.results['feature_comparison'] = results
        self._save_results('feature_comparison', results)
        
        logger.info("Completed feature set comparison")
        return results
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all four research objectives.
        
        Returns:
            Dict[str, Any]: Complete results from all experiments
        """
        logger.info("Starting all SEED experiments...")
        start_time = time.time()
        
        # Run all experiments
        freq_results = self.run_frequency_band_analysis()
        channel_results = self.run_channel_analysis()
        classifier_results = self.run_classifier_comparison()
        feature_results = self.run_feature_comparison()
        
        # Compile complete results
        complete_results = {
            'frequency_analysis': freq_results,
            'channel_analysis': channel_results,
            'classifier_comparison': classifier_results,
            'feature_comparison': feature_results,
            'experiment_info': {
                'total_runtime': time.time() - start_time,
                'config': self.config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save complete results
        self._save_results('complete_results', complete_results)
        
        logger.info(f"Completed all experiments in {time.time() - start_time:.2f} seconds")
        return complete_results
    
    def _save_results(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """
        Save experiment results to JSON file.
        
        Args:
            experiment_name (str): Name of the experiment
            results (Dict[str, Any]): Results to save
        """
        output_file = self.results_dir / f"{experiment_name}.json"
        
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
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Saved {experiment_name} results to {output_file}")
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for all experiments.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        summary_data = []
        
        for experiment_name, experiment_results in self.results.items():
            for condition, condition_results in experiment_results.items():
                if isinstance(condition_results, dict) and 'mean_accuracy' in condition_results:
                    summary_data.append({
                        'Experiment': experiment_name,
                        'Condition': condition,
                        'Mean_Accuracy': condition_results['mean_accuracy'],
                        'Std_Accuracy': condition_results['std_accuracy'],
                        'N_Repeats': len(condition_results['all_accuracies'])
                    })
        
        return pd.DataFrame(summary_data)