# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: log_viewer.py
# Date: 2025/10/10 11:27:08
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


import pymongo
import streamlit as st


def ErrorViewer():
    collection_description = {
	    "CAP6614_MARINE": "CAP 6614: MARINE inference logs",
    }
    st.header("Error Viewer", divider="blue")
    



    mongo = st.session_state['mongo_client']
    chosen_collection = st.selectbox(
        "Select Task",
        options=list(collection_description.keys()),
        format_func=lambda x: collection_description[x],
    )


    if st.button("Show errors", use_container_width=True):
        data = pd.DataFrame.from_records(mongo['LOGS'][chosen_collection].find().to_list()).sort_values(by='timestamp', ascending=False).drop(columns=['_id']).head(1000)

        # Assign a color to each unique error type
        unique_errors = data['error'].unique()
        color_map = {err: f'background-color: {color}; color: black' for err, color in zip(unique_errors, 
                ['#8ac0e5', '#e5afcf', '#f7e4a6', '#ffffcc', '#bcebe4', '#a497c4', '#a42384', '#ffecd4', '#ffe4e1', '#d3ffd3'])}

        def highlight_error(row):
            color = color_map.get(row['error'], '')
            return [color] * len(row) if color else [''] * len(row)

        styled_data = data.style.apply(highlight_error, axis=1)
        st.dataframe(styled_data, use_container_width=True, height=1000)
