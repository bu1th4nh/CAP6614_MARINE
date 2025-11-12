# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: main_marine_streamlit_app.py
# Date: 2025/11/05 16:59:19
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication at the University of Central Florida, EPCOT, and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------


import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Literal

from s3fs import S3FileSystem
import streamlit as st


import base64
import uuid
import pymongo
import streamlit as st


from gui_utils.log_viewer import ErrorViewer
from gui_utils.answer_viewer import AnswerViewer

# -----------------------------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------------------------
st.set_page_config(
    page_title = "MARINE Helper System",
    page_icon = "⚔️",
    layout = "wide"
)
st.title('MARINE Helper System')

# -----------------------------------------------------------------------------------------------
# MongoDB Connection
# -----------------------------------------------------------------------------------------------
mongo = pymongo.MongoClient(
    host='mongodb://localhost',
    port=27017,
    username='bu1th4nh',
    password='ariel.anna.elsa',
)



# -----------------------------------------------------------------------------------------------
# MinIO
# -----------------------------------------------------------------------------------------------
key = 'bu1th4nh'
secret = 'ariel.anna.elsa'
endpoint_url = 'http://localhost:9000'

s3 = S3FileSystem(
    anon=False, 
    endpoint_url=endpoint_url,
    key=key,
    secret=secret,
    use_ssl=False
)
storage_options = {
    'key': key,
    'secret': secret,
    'endpoint_url': endpoint_url,
}





# -----------------------------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------------------------
with st.spinner("Loading Data..."):
    st.session_state['s3'] = s3
    st.session_state['storage_options'] = storage_options
    st.session_state['mongo_client'] = mongo
    

tabs = st.tabs(["Error Viewer", "Answer Viewer"])

with tabs[0]: 
    ErrorViewer()

with tabs[1]:
    AnswerViewer()