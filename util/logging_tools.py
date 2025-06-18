import fus_anes.config as config
import logging

def setup_logging():
    logger = logging.getLogger()
    logging.captureWarnings(True)
    logger.setLevel(logging.NOTSET)
    handler_file = logging.FileHandler(config.logging_file)
    handler_file.setLevel(logging.NOTSET)
    handler_stdout = logging.StreamHandler()
    handler_stdout.setLevel(logging.NOTSET)
    
    formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%Y.%m.%d-%H.%M.%S')
    handler_file.setFormatter(formatter)
    handler_stdout.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stdout)
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.info('Logging setup, beginning run.')
