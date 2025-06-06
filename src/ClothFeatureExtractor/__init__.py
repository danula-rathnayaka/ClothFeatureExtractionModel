import os
import sys
import logging
from datetime import datetime
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "../../../logs"
log_filepath = os.path.join(log_dir, f"{datetime.now().strftime("%Y%m%d_%H%M%S")}.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Logger")
path_to_root = Path("../../../")
