"""
Utility functions for SEED EEG emotion recognition project.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str): Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (only warnings and errors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (all messages)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def check_for_infinite_loops(max_iterations: int = 1000):
    """
    Decorator to prevent infinite loops by limiting iterations.
    
    Args:
        max_iterations (int): Maximum allowed iterations
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            iteration_count = 0
            
            def count_iteration():
                nonlocal iteration_count
                iteration_count += 1
                if iteration_count > max_iterations:
                    raise RuntimeError(f"Function {func.__name__} exceeded maximum iterations ({max_iterations})")
            
            # Inject iteration counter into function if it accepts it
            if 'iteration_counter' in func.__code__.co_varnames:
                kwargs['iteration_counter'] = count_iteration
            
            return func(*args, **kwargs)
        return wrapper
    return decorator