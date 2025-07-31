import logging

def setup_logger(name: str = __name__, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with both console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level (default=logging.INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
