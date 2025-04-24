import logging
import os


def setup_logger(name, log_file="app.log"):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger