"""Logging utilities shared across MAEM scripts."""

import logging
import os
import sys
from datetime import datetime
from typing import Tuple


def setup_logger(output_dir: str, log_name: str = 'evaluation') -> Tuple[logging.Logger, str]:
    """Setup logger to write to both file and console."""
    logger = logging.getLogger('multiview_eval')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'{log_name}_{timestamp}.log')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger, log_file
