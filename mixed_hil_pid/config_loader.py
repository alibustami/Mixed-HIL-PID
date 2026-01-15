"""
Configuration loader utility for HIL PID optimization.

This module provides a simple interface to load configuration from config.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default: config.yaml in project root
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_pid_bounds(config: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Get PID bounds as list of tuples.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of (min, max) tuples for [Kp, Ki, Kd]
    """
    return [
        tuple(config['pid_bounds']['kp']),
        tuple(config['pid_bounds']['ki']),
        tuple(config['pid_bounds']['kd'])
    ]
