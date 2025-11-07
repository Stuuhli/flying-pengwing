import mimetypes
import os
import base64
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'utils/dev_frontend.env'))

API_CONV_START= "http://localhost:8000//conversation/start"
API_VALIDATE_USER= "http://localhost:8000/validate_user/{user}"
API_CONV_SEND=  "http://localhost:8000/conversation/{session_id_user}/message"
API_GET_HISTORY= "http://localhost:8000/get_conversation/{session_id_user}"
API_LOGOUT= "http://localhost:8000/logout"
API_GET_EXISTING_CONV= "http://localhost:8000/get_existing_conv_ids/{session_id_user}"
API_GET_AVAILABLE_DOC_NAMES="http://localhost:8000/get_user_available_docs_check_admin/{session_id_user}"
API_INGEST_DOC_FRONTEND="http://localhost:8000/ingest_doc_frontend/{session_id_user}/file_name"
API_CREATE_EMPTY_COLLECTION= "http://localhost:8000/create_collection/{collection_name}"
MILVUS_URI= "http://localhost:19530"
API_LOG_FEEDBACK= "http://localhost:8000/log_feedback/"

# logo at login
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
ICON_PATH = parent_dir + os.environ.get("ICON_PATH")
mime, _ = mimetypes.guess_type(ICON_PATH)
with open(ICON_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
data_uri = f"data:{mime};base64,{b64}"
auth_html=  f"""
<div style="text-align:center; margin-top:-12px; padding:0;">
  <div style="width:256px; height:96px; margin:0 auto; display:flex; align-items:center; justify-content:center;">
    <img src="{data_uri}" style="max-width:100%; max-height:100%; object-fit:contain; margin:0;" />
  </div>
  <p style='font-size: 24px;'>Welcome to <b>KI-Pilot - AI Assistant<b></p>
</div>
"""