import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
from gradio import ChatMessage  # noqa: E402
import copy # noqa: E402
import gradio as gr # noqa: E402
import requests # noqa: E402
import uuid # noqa: E402
from config_URL import API_VALIDATE_USER, API_CONV_START, API_CONV_SEND, API_GET_HISTORY, API_LOGOUT, API_INGEST_DOC_FRONTEND, API_CREATE_EMPTY_COLLECTION, auth_html # noqa: E402
from utils.utils_helper_func import show_client, get_all_sessions, convert_chat, get_doc_names_frontend, feedback_logger # noqa: E402
from utils.utils_logging import logger, initialize_logging, FRONTEND_LOG # noqa: E402

# logging config
initialize_logging(FRONTEND_LOG)

# Session state management
class SessionState:
    def __init__(self):
        self.authenticated = False
        self.session_id = ""
        self.username = ""
        self.read_collection = ""
        self.ingest_collection=""
        self.user_client = None
        self.existing_conversations=[]

session_instances={}

# Gradio functions
def initialize_instance(request: gr.Request):
    """ Update global session variable dict with session hash id as the key for a user state for better downstream processing.
        Send data to frontend to display: Username, existing sessions dropdown, available documents to chat with

    Args:
        request (gr.Request): session request when user opens the frontend url

    Returns: Markdown with user name after login
    """
    if request.username in session_instances.keys():
        session_instances[request.session_hash] = session_instances.pop(request.username)
        logger.info("Session id: %s initialised for user: %s", session_instances[request.session_hash].session_id, request.username)
        existing_choices= session_instances[request.session_hash].existing_conversations
        conv_id=session_instances[request.session_hash].session_id
        available_doc_markdown, admin_access, col_type = get_doc_names_frontend(conv_id= conv_id, username= request.username)
        ingestion_button_updated = gr.UploadButton(label="Ingest another document", size="md", variant="primary", file_types=["file"], interactive= admin_access)
        return f"# KI-Pilot - AI Assistant (Mode: {col_type})<br>", f"# KI-Pilot - AI Assistant (Mode: {col_type})", gr.Dropdown(choices=existing_choices, label="Select Conversation", value=None), gr.Dataframe(available_doc_markdown), ingestion_button_updated

def upload_file(filepath, request: gr.Request):
    ''' Get collection from read collection which itself gets its value from user_collection mapping: thus user can only upload to the collection they are assigned to if they are admin'''
    if not session_instances[request.session_hash].authenticated:
        return "Please authenticate first."
    logger.info("User %s ingesting file: %s to collection: %s", request.username, filepath.split("/")[-1], session_instances[request.session_hash].read_collection)
    data = {"conv_id": session_instances[request.session_hash].session_id, "file": filepath, "ingest_collection":session_instances[request.session_hash].read_collection}
    try:
        response = requests.post(
            API_INGEST_DOC_FRONTEND.format(session_id_user=f"{session_instances[request.session_hash].session_id}${session_instances[request.session_hash].username}"),
            json=data
        )
        response.raise_for_status()
        message= response.json()[1]
    except Exception as e:
        logger.error("Got an error during uploading file: %s for session id: %s: %s", filepath.split("/")[-1], session_instances[request.session_hash].session_id, str(e))
    
    available_doc_markdown, admin_access, _ = get_doc_names_frontend(conv_id= session_instances[request.session_hash].session_id, username= request.username)
    upload_button_updated = gr.UploadButton(label="Ingest another document", size="md", variant="primary", file_types=["file"], interactive= admin_access)
    return upload_button_updated, gr.Markdown(message, visible=True), gr.Dataframe(available_doc_markdown)

def send_chat(message, request:gr.Request):
    """ Send user message to chat api and get response

    Args:
        message (str): user_message
        request (gr.Request): requests from user session

    Returns:
        reply: reply from chatbot
    """
    if not session_instances[request.session_hash].authenticated:
        return "Please authenticate first."
    data = {"conv_id": session_instances[request.session_hash].session_id, "message": message}
    try:
        response = requests.post(
            API_CONV_SEND.format(session_id_user=f"{session_instances[request.session_hash].session_id}${session_instances[request.session_hash].username}"),
            json=data
        )
        response.raise_for_status()
        reply = str(response.json())
        logger.info("Bot response successful for user: %s for session id: %s", session_instances[request.session_hash].username, session_instances[request.session_hash].session_id)
        return reply
    except Exception as e:
        logger.error("Got an error during getting bot response for session id: %s: %s",session_instances[request.session_hash].session_id, str(e))
        return f"Chat error: {str(e)}"

def respond(message, chat_history, request: gr.Request):
        if message:
            response= send_chat(message, request=request)
        else:
            response= "What can I help you with?"
        chat_history.append(ChatMessage(role="user", content=message))
        chat_history.append(ChatMessage(role="assistant", content=response))
        return "",chat_history

def handle_feedback(like_data: gr.LikeData, request:gr.Request):
    """Handle like/dislike events and show feedback input for dislikes"""
    conv_id= session_instances[request.session_hash].session_id
    username= session_instances[request.session_hash].username
    if not like_data.liked:  # If disliked
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), like_data.value[0]
    else:  # If liked
        feedback_response= feedback_logger(conv_id= conv_id, username= username, 
                        LLM_response=like_data.value[0], feedback= "Positive", feedback_comment= "N/A")
        if not feedback_response:
            logger.error("Feedback not logged for user %s in session: %s", username, conv_id)
        else: 
            gr.Info("Thank you for your feedback!")
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), like_data.value[0]
    
def submit_feedback(feedback_text, message_content, request:gr.Request):
    conv_id= session_instances[request.session_hash].session_id
    username= session_instances[request.session_hash].username
    """Process the feedback submitted by user"""
    if feedback_text.strip():
        feedback_response= feedback_logger(conv_id= conv_id, username= username, 
                        LLM_response=message_content, feedback= "Negative", feedback_comment= feedback_text)
        print(f"Feedback received for message '{message_content}': {feedback_text}")
        if not feedback_response: 
            logger.error("Feedback not logged for user %s in session: %s", username, conv_id)
        else: 
            logger.info("Feedback: %s submitted for user %s, session: %s", feedback_text, username, conv_id)
            gr.Info("Thank you for your feedback!")
    return "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def dropdown_update_convo(dropdown, evt: gr.SelectData, request: gr.Request):
    """ Function that gets triggered when user selects conv from drop down. It gets conversation history of selected session and updates session id to selected session for
        both frontend and backend states. Then it converts the chat element to gradio chat element and updates the window in frontend.

    Args:
        dropdown (_type_): dropdown function
        evt (gr.SelectData): Has the selected conv id value
        request (gr.Request): network variables containing username

    Returns:
        chatbot_selected: chat history of selected conv id
    """
    data= {"old_conv_id": session_instances[request.session_hash].session_id, "new_conv_id": str(evt.value), "username": request.username}
    response = requests.post(API_GET_HISTORY.format(session_id_user=str(evt.value) + "$" + request.username), json=data)
    response.raise_for_status()
    session_instances[request.session_hash].session_id=str(evt.value)
    logger.info("User %s changed chat session to previous session: %s", request.username, evt.value)
    chatbot_selected = gr.Chatbot(value=convert_chat(response.json()),type="messages", placeholder="Hi, how can I help you today?", scale=9, resizable=True, show_copy_button=True, height=500)
    return chatbot_selected

def authenticate_user(username: str, password: str):
    """ First validates authentication and then populates the temporary session varibales to init the backend for user. 
        Finally adds user state object to global dict

    Args:
        username (str): username
        password (str): password

    Returns:
        bool: Whether succesfully authenticated or not
    """
    data = {"username": username, "password": password}
    try:
        logger.info("Logging in for: %s", username)
        response = requests.post(API_VALIDATE_USER.format(user=username), json=data)
        response.raise_for_status()
        success = response.json()[1]
        if success:
            temp_state=SessionState()
            temp_state.authenticated = True
            temp_state.username = username
            temp_state.session_id = str(uuid.uuid4())
            temp_state.user_client = show_client(username=username, password=password)
            collections = temp_state.user_client.list_collections()
            init_data = {
                "conv_id": temp_state.session_id,
                "username": temp_state.username,
                "password": password
            }
            init_response = requests.post(API_CONV_START, json=init_data)
            init_response.raise_for_status()
            logger.info("User %s has collection: %s assigned at the backend", username, init_response.json()["user_collection"])
            temp_state.read_collection= init_response.json()["user_collection"]
            # create collection if not present
            if temp_state.read_collection not in collections:
                logger.warning("No collections exist for user: %s, creating collection %s", username, temp_state.read_collection)
                collection_response= requests.post(API_CREATE_EMPTY_COLLECTION.format(collection_name= temp_state.read_collection))
                collection_response.raise_for_status()
                if not collection_response.json():
                    return False
            temp_state.existing_conversations= get_all_sessions(username=username, new_conv_id= copy.deepcopy(temp_state.session_id))
            session_instances[username]=copy.deepcopy(temp_state)
            logger.info("Current object created for %s has existing convos: %s", username, temp_state.existing_conversations)
            logger.info("Successful authentication for %s, session id: %s", username, temp_state.session_id)
            return True
        else:
            logger.info("Unsuccessful authentication for %s",username)
            return False
    except Exception as e:
        logger.error("Got an error during authentication: %s", str(e))
        return False

def do_quit(request: gr.Request):
        """Exit the chat
        """
        data= {
            "conv_id": session_instances[request.session_hash].session_id,
            "user": request.username
        }
        response= requests.post(API_LOGOUT, json=data)
        if response.status_code==200:
            logger.info("%s for user %s", response.json(), request.username)
        else:
            logger.error("Error occurred: %s", response.json())
            raise Exception("Error while logging out")


def cleanup_instance(request: gr.Request):
    if request.session_hash in session_instances:
        logger.info("User %s logged out", request.username)
        del session_instances[request.session_hash]
        logger.debug("Current global session cache: %s", session_instances.keys())

""" Gradio Interface Setup using Blocks API (for flexibility) """
with gr.Blocks() as chat:
    chatbot = gr.Chatbot(type="messages", placeholder="Hi, how can I help you today?", scale=15, resizable=True, show_copy_button=True, height=600, container=False, sanitize_html=False, feedback_options=("Like", "Dislike"))
    msg = gr.Textbox(type="text", placeholder="Type your question here...", label="User Message", submit_btn=True, scale=15)
    clear = gr.ClearButton([msg, chatbot], size="sm")
    last_message = gr.State("")
    msg.submit(respond, [msg, chatbot], [msg, chatbot], trigger_mode="once")
    with gr.Row():
        feedback_input = gr.Textbox(
            placeholder="What went wrong? Your feedback would be crucial in improving the system.", 
            label="Feedback", visible=False, lines=3, autofocus=True, autoscroll=True
            )
    with gr.Row():
        submit_feedback_btn = gr.Button("Submit Feedback", visible=False, variant="primary")
        cancel_feedback_btn = gr.Button("Cancel", visible=False)

with gr.Blocks(fill_width=True) as demo:
    with gr.Tab("Chat_Session"):
        m= gr.Markdown("# KI-Pilot - AI Assistant")
        conversation_list=[]
        chat.render()
        with gr.Sidebar(position="left", width=360, open=False):
            gr.Markdown("### Widgets")
            select_conversation = gr.Dropdown(
                choices=conversation_list, 
                label="Select Conversation",
                value=None
            )
            gr.HTML("<div style='flex-grow: 1;'></div>")
            logout_button_front = gr.Button("Logout", size="sm")
    with gr.Tab("Ingested Documents (RAG)"):
        doc_markdown= gr.Markdown("# KI-Pilot - AI Assistant")
        db= gr.Dataframe()
        ingestion_button = gr.UploadButton(label="Ingest another document", size="md", variant="primary", file_types=["file"], interactive= True)
        gr.Markdown("Note: Only PDFs < 50 MB are supported!")
        result= gr.Markdown(visible=False)
        with gr.Sidebar(position="left", width=360, open=False):
            gr.Markdown("### Widgets")
            logout_button_back = gr.Button("Logout", size="sm")
    demo.load(initialize_instance, None, [m, doc_markdown, select_conversation, db, ingestion_button], concurrency_limit=20)

    chatbot.like(handle_feedback, 
        None, 
        [feedback_input, submit_feedback_btn, cancel_feedback_btn, msg, clear, last_message])
    
    submit_feedback_btn.click(
        submit_feedback,
        [feedback_input, last_message],
        [feedback_input, feedback_input, submit_feedback_btn, cancel_feedback_btn, msg, clear]
    )

    cancel_feedback_btn.click(
        lambda: ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                  gr.update(visible=True), gr.update(visible=True)),
        None,
        [feedback_input, feedback_input, submit_feedback_btn, cancel_feedback_btn, 
         msg, clear]
    )

    ingestion_button.upload(fn=lambda x: gr.update(interactive=False), 
        inputs=ingestion_button, 
        outputs=ingestion_button
    ).then(fn= upload_file, inputs=ingestion_button, outputs=[ingestion_button, result, db], 
           trigger_mode= "once", show_progress="full", show_progress_on= [ingestion_button, result, db])

    logout_button_front.click(do_quit, inputs=None, outputs=None).then(fn=None, inputs=None, outputs=None, js="() => { window.location.href = '/logout'; }")
    logout_button_back.click(do_quit, inputs=None, outputs=None).then(fn=None, inputs=None, outputs=None, js="() => { window.location.href = '/logout'; }")
    select_conversation.select(dropdown_update_convo, select_conversation, [chatbot])
    demo.unload(cleanup_instance)

demo.launch(share=False, auth=authenticate_user, auth_message=auth_html, show_error=True, server_name="0.0.0.0", server_port=8083)
