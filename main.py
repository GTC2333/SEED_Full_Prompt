#!/usr/bin/env python3
"""
SEED EEG Emotion Recognition - Main Execution Script

This script runs the complete SEED EEG emotion recognition research project,
implementing all four research objectives:

1. Frequency band analysis
2. Channel montage optimization  
3. Classifier comparison
4. Feature set comparison

Usage:
    python main.py [--config config/config.yaml] [--quick-test]
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from experiments import SEEDExperiments
from visualization import SEEDVisualizer

def setup_logging(config: dict) -> None:
    """
    Setup logging configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def modify_config_for_quick_test(config: dict) -> dict:
    """
    Modify configuration for quick testing with reduced parameters.
    
    Args:
        config (dict): Original configuration
        
    Returns:
        dict: Modified configuration for quick testing
    """
    # Reduce number of subjects and repeats for quick testing
    config['data']['n_subjects'] = 3
    config['experiment']['n_repeats'] = 2
    config['experiment']['train_sessions'] = 3
    config['experiment']['test_sessions'] = 2
    
    # Reduce epochs for deep learning
    config['models']['dbn']['n_epochs'] = 20
    
    # Use fewer frequency bands and features for testing
    config['data']['frequency_bands'] = {
        'alpha': [8, 13],
        'beta': [14, 30]
    }
    
    config['features']['types'] = ['de', 'psd']
    config['models']['classifiers'] = ['svm', 'logistic']
    
    logging.info("Configuration modified for quick testing")
    return config

def print_research_objectives():
    """Print the research objectives to console."""
    objectives = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    SEED EEG Emotion Recognition Research                     ║
    ║                              Research Objectives                             ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  1. FREQUENCY BAND ANALYSIS                                                  ║
    ║     • Identify which EEG frequency bands carry the most discriminative       ║
    ║       information for three-class emotion recognition                        ║
    ║     • Test: delta, theta, alpha, beta, gamma, and combinations               ║
    ║                                                                              ║
    ║  2. CHANNEL MONTAGE OPTIMIZATION                                             ║
    ║     • Find minimal electrode montage that retains emotion-discrimination     ║
    ║       capability equivalent to full 62-channel setup                        ║
    ║     • Test: 4, 6, 9, 12, and 62 channel configurations                      ║
    ║                                                                              ║
    ║  3. CLASSIFIER ARCHITECTURE COMPARISON                                       ║
    ║     • Determine whether deep architecture offers measurable advantage        ║
    ║       over shallow methods for EEG-based emotion recognition                 ║
    ║     • Test: SVM, Logistic Regression, k-NN, Deep Belief Network             ║
    ║                                                                              ║
    ║  4. FEATURE SET COMPARISON                                                   ║
    ║     • Establish which EEG feature set conveys the most discriminative        ║
    ║       information for emotion classification                                 ║
    ║     • Test: PSD, DE, DASM, RASM, DCAU, Combined Asymmetry                   ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(objectives)

def print_progress_header(objective_num: int, title: str):
    """Print progress header for each research objective."""
    header = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  RESEARCH OBJECTIVE {objective_num}: {title:<58} ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='SEED EEG Emotion Recognition Research')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    parser.add_argument('--objective', type=int, choices=[1, 2, 3, 4],
                       help='Run specific research objective only (1-4)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Modify config for quick testing if requested
    if args.quick_test:
        config = modify_config_for_quick_test(config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Print research objectives
    print_research_objectives()
    
    # Initialize experiment runner
    logger.info("Initializing SEED EEG emotion recognition experiments...")
    experiments = SEEDExperiments(config)
    
    # Initialize visualizer
    visualizer = SEEDVisualizer(config, config['data']['output_dir'])
    
    # Record start time
    start_time = time.time()
    
    try:
        if args.objective:
            # Run specific objective
            if args.objective == 1:
                print_progress_header(1, "FREQUENCY BAND ANALYSIS")
                results = experiments.run_frequency_band_analysis()
                visualizer.plot_frequency_band_analysis(results)
                
            elif args.objective == 2:
                print_progress_header(2, "CHANNEL MONTAGE OPTIMIZATION")
                results = experiments.run_channel_analysis()
                visualizer.plot_channel_analysis(results)
                
            elif args.objective == 3:
                print_progress_header(3, "CLASSIFIER ARCHITECTURE COMPARISON")
                results = experiments.run_classifier_comparison()
                visualizer.plot_classifier_comparison(results)
                
            elif args.objective == 4:
                print_progress_header(4, "FEATURE SET COMPARISON")
                results = experiments.run_feature_comparison()
                visualizer.plot_feature_comparison(results)
        
        else:
            # Run all experiments
            logger.info("Starting complete SEED EEG emotion recognition research...")
            
            print_progress_header(1, "FREQUENCY BAND ANALYSIS")
            freq_results = experiments.run_frequency_band_analysis()
            
            print_progress_header(2, "CHANNEL MONTAGE OPTIMIZATION")
            channel_results = experiments.run_channel_analysis()
            
            print_progress_header(3, "CLASSIFIER ARCHITECTURE COMPARISON")
            classifier_results = experiments.run_classifier_comparison()
            
            print_progress_header(4, "FEATURE SET COMPARISON")
            feature_results = experiments.run_feature_comparison()
            
            # Compile all results
            all_results = {
                'frequency_analysis': freq_results,
                'channel_analysis': channel_results,
                'classifier_comparison': classifier_results,
                'feature_comparison': feature_results,
                'experiment_info': {
                    'total_runtime': time.time() - start_time,
                    'config': config,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Create all visualizations
            logger.info("Creating comprehensive visualizations...")
            visualizer.create_all_visualizations(all_results)
            
            # Generate summary statistics
            summary_stats = experiments.get_summary_statistics()
            summary_file = Path(config['data']['output_dir']) / 'summary_statistics.csv'
            summary_stats.to_csv(summary_file, index=False)
            logger.info(f"Saved summary statistics to {summary_file}")
            
            # Print final summary
            total_time = time.time() - start_time
            print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           EXPERIMENT COMPLETED                               ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Total Runtime: {total_time/60:.1f} minutes                                           ║
    ║                                                                              ║
    ║  Results saved to: {config['data']['output_dir']:<50} ║
    ║  Visualizations saved to: {config['data']['output_dir']:<43} ║
    ║                                                                              ║
    ║  Key Findings:                                                               ║
    ║  • Best frequency band: {max(freq_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]:<48} ║
    ║  • Optimal channel count: {max(channel_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]:<44} ║
    ║  • Best classifier: {max(classifier_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]:<52} ║
    ║  • Best feature set: {max(feature_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]:<49} ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
            """)
    
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        raise
    
    logger.info("SEED EEG emotion recognition research completed successfully!")

if __name__ == "__main__":
    main()