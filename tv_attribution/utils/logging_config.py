import logging
import sys

def setup_logging(name="tv_attribution_logger", level=logging.INFO):
    """Sets up a production-standard logger for the TV Attribution pipeline."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # File Handler
    file_handler = logging.FileHandler("tv_attribution.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
