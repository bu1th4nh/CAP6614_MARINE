# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: log_config.py
# Date: 2025/10/31 15:25:51
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication at the University of Central Florida, EPCOT, and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------




import numpy as np
import pandas as pd

import sys
import logging
from typing import Union
from colorlog import ColoredFormatter

# -----------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------
def initialize_logging(log_filename: Union[str, None] = None):
    # logging.root = logging.getLogger('uvicorn.default')
    logging.root.handlers = [];

    # Console handler
    handler_sh = logging.StreamHandler(sys.stdout)
    handler_sh.setFormatter(
        ColoredFormatter(
            "%(cyan)s%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(light_white)s%(message)s%(reset)s %(blue)s(%(filename)s:%(lineno)d)",
            datefmt  = '%Y/%m/%d %H:%M:%S',
            log_colors={
                'DEBUG': 'white',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )

    logging.basicConfig(
        level    = logging.INFO,
        handlers = [handler_sh]
    )

    # handler_api_sh = logging.StreamHandler(sys.stdout)
    # handler_api_sh.setFormatter(
    #     ColoredFormatter(
    #         "%(light_purple)s%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(white)s[UVICORN]%(reset)s %(light_purple)s%(message)s",
    #         datefmt  = '%Y/%m/%d %H:%M:%S',
    #         log_colors={
    #             'DEBUG': 'gray',
    #             'INFO': 'green',
    #             'WARNING': 'yellow',
    #             'ERROR': 'red',
    #             'CRITICAL': 'red,bg_white',
    #         }
    #     )
    # )


    # File handler
    # if log_filename is not None:
    #     handler_file = logging.FileHandler(
    #         filename = log_filename,
    #         encoding = "utf-8-sig",    
    #         mode     = "w"
    #     )
    #     handler_file.setFormatter(
    #         logging.Formatter(
    #             fmt      = '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)', 
    #             datefmt  = '%Y/%m/%d %H:%M:%S'
    #         )
    #     )

    #     # Set logging level and handlers
    #     logging.basicConfig(
    #         level    = logging.INFO,
    #         handlers = [handler_sh, handler_file]
    #     )
    #     logging.getLogger("uvicorn").handlers = [handler_api_sh]
    # else:
    #     logging.basicConfig(
    #         level    = logging.INFO,
    #         handlers = [handler_sh]
    #     )
    #     logging.getLogger("uvicorn").handlers = [handler_api_sh]




