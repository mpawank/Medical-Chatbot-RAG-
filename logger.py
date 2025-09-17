import logging

def setup_logger(name="MedicalAssistant"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] === [%(message)s]")
    ch.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(ch)
    
    return logger


# Example usage
logger = setup_logger()

logger.info("RAG process started")
logger.debug("Debugging information")
logger.error("Failed to load")
logger.critical("Critical error")
