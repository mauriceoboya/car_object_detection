import logging
import os
from pathlib import Path

logs='logs'
log=os.path.join(logs,'logging.log')
if not Path(logs).exists():
    os.makedirs(logs, exist_ok=True)



project_name="CarObjectDetection"
   
FORMAT = '%(asctime)s  %(levelname)-8s %(message)s'
file_handler = logging.FileHandler(log)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(project_name)
