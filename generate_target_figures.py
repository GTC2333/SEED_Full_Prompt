#!/usr/bin/env python3
"""
Generate Target Figures Script

This script generates all the specific target figures requested for the SEED EEG research.
"""

import sys
import os
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from target_visualizations import TargetVisualizer
from utils import setup_logging

def main():
    """Main function to generate all target figures."""
    
    parser = argparse.ArgumentParser(description='Generate target figures for SEED EEG research')
    parser.add_argument('--output-dir', default='output/target_figures',
                       help='Output directory for figures')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', default='logs/target_figures.log',
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    print("🎯 Generating Target Figures for SEED EEG Research")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = TargetVisualizer(args.output_dir)
    
    try:
        # Generate all target figures
        print("📊 Creating classifier-feature comparison table...")
        visualizer.create_classifier_feature_comparison_table({})
        
        print("📈 Creating DE feature spectrogram...")
        visualizer.create_de_feature_spectrogram()
        
        print("🔍 Creating confusion matrices...")
        visualizer.create_confusion_matrices()
        
        print("⚖️ Creating DBN weight distribution plot...")
        visualizer.create_dbn_weight_distribution()
        
        print("🧠 Creating brain weight heatmap...")
        visualizer.create_brain_heatmap()
        
        print("📋 Creating comprehensive performance table...")
        visualizer.create_comprehensive_performance_table()
        
        print("\n✅ All target figures generated successfully!")
        print(f"📁 Figures saved to: {args.output_dir}")
        
        # List generated files
        output_path = Path(args.output_dir)
        if output_path.exists():
            print("\n📄 Generated files:")
            for file in sorted(output_path.glob("*")):
                if file.is_file():
                    print(f"  - {file.name}")
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()