"""
Visualization Module for SEED EEG Emotion Recognition

This module creates comprehensive visualizations for all research findings:
- Frequency band analysis plots
- Channel montage comparison plots
- Classifier performance comparisons
- Feature set comparison plots
- Statistical significance tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SEEDVisualizer:
    """
    Comprehensive visualization suite for SEED experiment results.
    """
    
    def __init__(self, config: Dict, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            config (Dict): Configuration dictionary
            output_dir (str): Output directory for saving plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        self.figure_size = config['visualization']['figure_size']
        self.dpi = config['visualization']['dpi']
        self.colors = config['visualization']['colors']
        self.save_formats = config['visualization']['save_formats']
        
        logger.info(f"Initialized visualizer with output directory: {output_dir}")
    
    def plot_frequency_band_analysis(self, results: Dict[str, Any]) -> None:
        """
        Create visualizations for frequency band analysis results.
        
        Args:
            results (Dict[str, Any]): Frequency band analysis results
        """
        logger.info("Creating frequency band analysis plots...")
        
        # Extract data for plotting
        conditions = list(results.keys())
        mean_accuracies = [results[cond]['mean_accuracy'] for cond in conditions]
        std_accuracies = [results[cond]['std_accuracy'] for cond in conditions]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bar plot with error bars
        bars = ax1.bar(range(len(conditions)), mean_accuracies, yerr=std_accuracies, 
                      capsize=5, alpha=0.8, color='steelblue')
        ax1.set_xlabel('Frequency Bands')
        ax1.set_ylabel('Classification Accuracy')
        ax1.set_title('EEG Frequency Band Analysis\nClassification Performance')
        ax1.set_xticks(range(len(conditions)))
        ax1.set_xticklabels(conditions, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accuracies, std_accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.01,
                    f'{mean_acc:.3f}±{std_acc:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Box plot showing distribution
        all_accuracies = [results[cond]['all_accuracies'] for cond in conditions]
        bp = ax2.boxplot(all_accuracies, labels=conditions, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Frequency Bands')
        ax2.set_ylabel('Classification Accuracy')
        ax2.set_title('Distribution of Accuracy Scores\nAcross Cross-Validation Repeats')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'frequency_band_analysis')
        plt.close()
        
        # Statistical significance test
        self._plot_statistical_significance(results, 'frequency_band_analysis')
    
    def plot_channel_analysis(self, results: Dict[str, Any]) -> None:
        """
        Create visualizations for channel montage analysis results.
        
        Args:
            results (Dict[str, Any]): Channel analysis results
        """
        logger.info("Creating channel montage analysis plots...")
        
        # Extract channel counts and accuracies
        channel_counts = []
        mean_accuracies = []
        std_accuracies = []
        
        for condition, result in results.items():
            if 'channels' in condition:
                n_channels = int(condition.split('_')[0])
                channel_counts.append(n_channels)
                mean_accuracies.append(result['mean_accuracy'])
                std_accuracies.append(result['std_accuracy'])
        
        # Sort by channel count
        sorted_data = sorted(zip(channel_counts, mean_accuracies, std_accuracies))
        channel_counts, mean_accuracies, std_accuracies = zip(*sorted_data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Line plot showing accuracy vs channel count
        ax1.errorbar(channel_counts, mean_accuracies, yerr=std_accuracies, 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.set_xlabel('Number of EEG Channels')
        ax1.set_ylabel('Classification Accuracy')
        ax1.set_title('Channel Montage Optimization\nAccuracy vs Number of Channels')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_xticks(channel_counts)
        ax1.set_xticklabels(channel_counts)
        
        # Add annotations
        for x, y, std in zip(channel_counts, mean_accuracies, std_accuracies):
            ax1.annotate(f'{y:.3f}±{std:.3f}', (x, y), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Bar plot for easier comparison
        bars = ax2.bar(range(len(channel_counts)), mean_accuracies, 
                      yerr=std_accuracies, capsize=5, alpha=0.8, color='green')
        ax2.set_xlabel('Channel Montage')
        ax2.set_ylabel('Classification Accuracy')
        ax2.set_title('Performance Comparison\nDifferent Channel Montages')
        ax2.set_xticks(range(len(channel_counts)))
        ax2.set_xticklabels([f'{n} ch' for n in channel_counts])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'channel_montage_analysis')
        plt.close()
    
    def plot_classifier_comparison(self, results: Dict[str, Any]) -> None:
        """
        Create visualizations for classifier comparison results.
        
        Args:
            results (Dict[str, Any]): Classifier comparison results
        """
        logger.info("Creating classifier comparison plots...")
        
        # Extract data
        classifiers = list(results.keys())
        mean_accuracies = [results[clf]['mean_accuracy'] for clf in classifiers]
        std_accuracies = [results[clf]['std_accuracy'] for clf in classifiers]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bar plot with error bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(range(len(classifiers)), mean_accuracies, yerr=std_accuracies,
                      capsize=5, alpha=0.8, color=colors[:len(classifiers)])
        ax1.set_xlabel('Classifier Type')
        ax1.set_ylabel('Classification Accuracy')
        ax1.set_title('Classifier Performance Comparison\nDeep vs Shallow Methods')
        ax1.set_xticks(range(len(classifiers)))
        ax1.set_xticklabels([clf.upper() for clf in classifiers])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accuracies, std_accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.01,
                    f'{mean_acc:.3f}±{std_acc:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Violin plot showing distributions
        all_accuracies = [results[clf]['all_accuracies'] for clf in classifiers]
        parts = ax2.violinplot(all_accuracies, positions=range(len(classifiers)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        ax2.set_xlabel('Classifier Type')
        ax2.set_ylabel('Classification Accuracy')
        ax2.set_title('Accuracy Distribution\nAcross Cross-Validation Repeats')
        ax2.set_xticks(range(len(classifiers)))
        ax2.set_xticklabels([clf.upper() for clf in classifiers])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'classifier_comparison')
        plt.close()
    
    def plot_feature_comparison(self, results: Dict[str, Any]) -> None:
        """
        Create visualizations for feature set comparison results.
        
        Args:
            results (Dict[str, Any]): Feature comparison results
        """
        logger.info("Creating feature set comparison plots...")
        
        # Extract data
        features = list(results.keys())
        mean_accuracies = [results[feat]['mean_accuracy'] for feat in features]
        std_accuracies = [results[feat]['std_accuracy'] for feat in features]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Horizontal bar plot for better label readability
        y_pos = np.arange(len(features))
        bars = ax1.barh(y_pos, mean_accuracies, xerr=std_accuracies,
                       capsize=5, alpha=0.8, color='purple')
        ax1.set_ylabel('Feature Type')
        ax1.set_xlabel('Classification Accuracy')
        ax1.set_title('Feature Set Comparison\nDiscriminative Information Content')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([feat.upper() for feat in features])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accuracies, std_accuracies)):
            width = bar.get_width()
            ax1.text(width + std_acc + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{mean_acc:.3f}±{std_acc:.3f}',
                    ha='left', va='center', fontsize=10)
        
        # Plot 2: Radar chart for feature comparison
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        mean_accuracies_radar = mean_accuracies + [mean_accuracies[0]]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, mean_accuracies_radar, 'o-', linewidth=2, color='purple')
        ax2.fill(angles, mean_accuracies_radar, alpha=0.25, color='purple')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([feat.upper() for feat in features])
        ax2.set_title('Feature Performance Radar Chart', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        self._save_figure(fig, 'feature_set_comparison')
        plt.close()
    
    def plot_comprehensive_summary(self, all_results: Dict[str, Any]) -> None:
        """
        Create a comprehensive summary plot of all experiments.
        
        Args:
            all_results (Dict[str, Any]): Results from all experiments
        """
        logger.info("Creating comprehensive summary plot...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Frequency bands (top left)
        ax1 = plt.subplot(2, 3, 1)
        freq_results = all_results['frequency_analysis']
        conditions = list(freq_results.keys())
        mean_accs = [freq_results[cond]['mean_accuracy'] for cond in conditions]
        std_accs = [freq_results[cond]['std_accuracy'] for cond in conditions]
        
        bars1 = ax1.bar(range(len(conditions)), mean_accs, yerr=std_accs, 
                       capsize=3, alpha=0.8, color='steelblue')
        ax1.set_title('Frequency Band Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(conditions)))
        ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Channel montages (top middle)
        ax2 = plt.subplot(2, 3, 2)
        channel_results = all_results['channel_analysis']
        
        channel_counts = []
        mean_accs = []
        std_accs = []
        for condition, result in channel_results.items():
            if 'channels' in condition:
                n_channels = int(condition.split('_')[0])
                channel_counts.append(n_channels)
                mean_accs.append(result['mean_accuracy'])
                std_accs.append(result['std_accuracy'])
        
        sorted_data = sorted(zip(channel_counts, mean_accs, std_accs))
        channel_counts, mean_accs, std_accs = zip(*sorted_data)
        
        ax2.errorbar(channel_counts, mean_accs, yerr=std_accs, 
                    marker='o', linewidth=2, markersize=6, capsize=3, color='green')
        ax2.set_title('Channel Montage Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Channels')
        ax2.set_ylabel('Accuracy')
        ax2.set_xscale('log')
        ax2.set_xticks(channel_counts)
        ax2.set_xticklabels(channel_counts)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Classifiers (top right)
        ax3 = plt.subplot(2, 3, 3)
        clf_results = all_results['classifier_comparison']
        classifiers = list(clf_results.keys())
        mean_accs = [clf_results[clf]['mean_accuracy'] for clf in classifiers]
        std_accs = [clf_results[clf]['std_accuracy'] for clf in classifiers]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars3 = ax3.bar(range(len(classifiers)), mean_accs, yerr=std_accs,
                       capsize=3, alpha=0.8, color=colors[:len(classifiers)])
        ax3.set_title('Classifier Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(range(len(classifiers)))
        ax3.set_xticklabels([clf.upper() for clf in classifiers], fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Features (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        feat_results = all_results['feature_comparison']
        features = list(feat_results.keys())
        mean_accs = [feat_results[feat]['mean_accuracy'] for feat in features]
        std_accs = [feat_results[feat]['std_accuracy'] for feat in features]
        
        y_pos = np.arange(len(features))
        bars4 = ax4.barh(y_pos, mean_accs, xerr=std_accs,
                        capsize=3, alpha=0.8, color='purple')
        ax4.set_title('Feature Set Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Accuracy')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([feat.upper() for feat in features], fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Overall best results (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        
        # Find best result from each experiment
        best_results = {
            'Frequency': max(freq_results.values(), key=lambda x: x['mean_accuracy']),
            'Channels': max(channel_results.values(), key=lambda x: x['mean_accuracy']),
            'Classifiers': max(clf_results.values(), key=lambda x: x['mean_accuracy']),
            'Features': max(feat_results.values(), key=lambda x: x['mean_accuracy'])
        }
        
        experiments = list(best_results.keys())
        best_accs = [best_results[exp]['mean_accuracy'] for exp in experiments]
        best_stds = [best_results[exp]['std_accuracy'] for exp in experiments]
        
        bars5 = ax5.bar(range(len(experiments)), best_accs, yerr=best_stds,
                       capsize=3, alpha=0.8, color='gold')
        ax5.set_title('Best Results Summary', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Best Accuracy')
        ax5.set_xticks(range(len(experiments)))
        ax5.set_xticklabels(experiments, fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc, std in zip(bars5, best_accs, best_stds):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 6: Statistical summary table (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary statistics table
        summary_data = []
        for exp_name, exp_results in all_results.items():
            if exp_name != 'experiment_info':
                for condition, result in exp_results.items():
                    if isinstance(result, dict) and 'mean_accuracy' in result:
                        summary_data.append([
                            exp_name.replace('_', ' ').title(),
                            condition.replace('_', ' ').title(),
                            f"{result['mean_accuracy']:.3f}",
                            f"{result['std_accuracy']:.3f}"
                        ])
        
        # Show only top 10 results
        summary_data.sort(key=lambda x: float(x[2]), reverse=True)
        summary_data = summary_data[:10]
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Experiment', 'Condition', 'Mean Acc', 'Std Acc'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Top 10 Results', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('SEED EEG Emotion Recognition - Comprehensive Results Summary', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        self._save_figure(fig, 'comprehensive_summary')
        plt.close()
    
    def _plot_statistical_significance(self, results: Dict[str, Any], experiment_name: str) -> None:
        """
        Create statistical significance analysis plot.
        
        Args:
            results (Dict[str, Any]): Experiment results
            experiment_name (str): Name of the experiment
        """
        conditions = list(results.keys())
        n_conditions = len(conditions)
        
        # Create pairwise comparison matrix
        p_values = np.ones((n_conditions, n_conditions))
        
        for i in range(n_conditions):
            for j in range(i+1, n_conditions):
                # Perform t-test between conditions
                data1 = results[conditions[i]]['all_accuracies']
                data2 = results[conditions[j]]['all_accuracies']
                
                _, p_value = stats.ttest_ind(data1, data2)
                p_values[i, j] = p_value
                p_values[j, i] = p_value
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Mask diagonal
        mask = np.eye(n_conditions, dtype=bool)
        
        sns.heatmap(p_values, mask=mask, annot=True, fmt='.3f', 
                   xticklabels=conditions, yticklabels=conditions,
                   cmap='RdYlBu_r', center=0.05, ax=ax)
        
        ax.set_title(f'Statistical Significance Matrix\n{experiment_name.replace("_", " ").title()}')
        plt.tight_layout()
        
        self._save_figure(fig, f'{experiment_name}_significance')
        plt.close()
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure in multiple formats.
        
        Args:
            fig (plt.Figure): Figure to save
            filename (str): Base filename
        """
        for fmt in self.save_formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.debug(f"Saved figure: {filepath}")
    
    def create_all_visualizations(self, all_results: Dict[str, Any]) -> None:
        """
        Create all visualizations for the complete experiment results.
        
        Args:
            all_results (Dict[str, Any]): Complete results from all experiments
        """
        logger.info("Creating all visualizations...")
        
        # Individual experiment plots
        if 'frequency_analysis' in all_results:
            self.plot_frequency_band_analysis(all_results['frequency_analysis'])
        
        if 'channel_analysis' in all_results:
            self.plot_channel_analysis(all_results['channel_analysis'])
        
        if 'classifier_comparison' in all_results:
            self.plot_classifier_comparison(all_results['classifier_comparison'])
        
        if 'feature_comparison' in all_results:
            self.plot_feature_comparison(all_results['feature_comparison'])
        
        # Comprehensive summary
        self.plot_comprehensive_summary(all_results)
        
        logger.info("Completed all visualizations")