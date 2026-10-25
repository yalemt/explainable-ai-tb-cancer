#!/usr/bin/env bash
# Local launch (no internet needed after install):  pip install -r requirements.txt && ./run.sh
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run app.py --server.address 127.0.0.1
