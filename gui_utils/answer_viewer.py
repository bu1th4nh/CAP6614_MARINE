# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: answer_viewer.py
# Date: 2025/11/05 17:01:26
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

import streamlit as st
import json

def AnswerViewer():
    st.header("Answer Viewer", divider="blue")

    path = "s3://results/CAP6614_MARINE/answers/instructblip"
    options = st.session_state['s3'].ls(path, detail=False)

    chosen_file = st.selectbox(
        "Select Answer File",
        options=options,
    )

    if st.button("Load Answer", use_container_width=True):
        # Read the selected file from S3 as lines
        lines = st.session_state['s3'].cat(chosen_file).splitlines()
        data = pd.DataFrame([json.loads(line) for line in lines])
        # Apply pink background to all cells using Streamlit's dataframe styling
        data = data[['image_id', 'prompt', 'text']].style.applymap(lambda v: 'background-color: lightblue; color: black')

        st.dataframe(data, use_container_width=True, height=1000)