import logging.config
import yaml
import os
from app.config import BACKEND_FASTAPI_LOG
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)


def setup_logging(log_file_path: str):
    """ Make the config from yaml file as logging config

    Args:
        log_file_path (str): Path where fastapi logs will be stored
    """
    with open(parent_dir+'/logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['handlers']['file']['filename'] = log_file_path
    logging.config.dictConfig(config)

# Get the logger
logger = logging.getLogger('timing_middleware')

def initialize_logging(log_path= BACKEND_FASTAPI_LOG):
    setup_logging(log_path)