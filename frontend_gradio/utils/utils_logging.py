import logging.config
import yaml
import os
from dotenv import load_dotenv
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
parent_dir_home= os.path.dirname(parent_dir)
# Load .env variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'dev_frontend.env'))

COMPUTE_MODE = os.getenv("COMPUTE_MODE")

if COMPUTE_MODE=="NON_SLURM":
    FRONTEND_LOG= parent_dir_home + os.getenv("FRONTEND_LOG")
else:
    FRONTEND_LOG= "." + os.getenv("FRONTEND_LOG")

def setup_logging(log_file_path: str):
    """ Make the config from yaml file as logging config

    Args:
        log_file_path (str): Path where fastapi logs will be stored
    """
    with open(parent_dir_home+ '/frontend_gradio/frontend_logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['handlers']['file']['filename'] = log_file_path
    logging.config.dictConfig(config)

# Get the logger
logger = logging.getLogger('gradio_logger')

def initialize_logging(log_path= FRONTEND_LOG):
    setup_logging(log_path)