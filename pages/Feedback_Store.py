import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from googleDriveUpload import load_json_from_google_drive,list_files_in_google_drive_folder,show_feedback
import streamlit as st
import pandas as pd
import numpy as np
import json
import time

st.set_page_config(page_title = 'Feedback Store')
st.header('A page used for showing submitted feedback. Here you can view feedback given to the LLM app.')

with st.spinner('Downloading Feedback Data...'):
    _, df = show_feedback()
    st.dataframe(df)
    


