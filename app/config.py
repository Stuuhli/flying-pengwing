from dotenv import load_dotenv
import os
from fastapi import HTTPException
from app.prompt_config import systemprompt_product, systemprompt_support, systemprompt_compliance

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# Load .env variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'dev.env'))

# Session specific Databases
USER_HISTORY = parent_dir+ os.environ.get("USER_HISTORY")
CHAT_STORE_PATH = parent_dir + os.environ.get("CHAT_STORE_PATH")
USER_DB_PATH = parent_dir + os.environ.get("USER_DB_PATH")
USER_COLLECTION_MAPPING= parent_dir + os.environ.get("USER_COLLECTION_MAPPING")

# Observability Databases
RETRIEVAL_LOG_PATH= parent_dir + os.environ.get("RETRIEVAL_LOG_PATH")
BACKEND_FASTAPI_LOG= parent_dir + os.environ.get("BACKEND_FASTAPI_LOG")

# API URLS
FASTAPI_PORT= os.environ.get("FASTAPI_PORT")
FASTAPI_URL= f"http://127.0.0.1:{FASTAPI_PORT}"
API_CONV_START = os.environ.get("API_CONV_START") # initialize conversation
API_CONV_SEND = os.environ.get("API_CONV_SEND") # send message to bot
API_GET_HISTORY = os.environ.get("API_GET_HISTORY") # get history of a conversation id
API_CREATE_USER = os.environ.get("API_CREATE_USER") # create new user
API_VALIDATE_USER = os.environ.get("API_VALIDATE_USER") # validate auth
#API_FILE_INGEST = os.environ.get("API_FILE_INGEST") # ingest file to vector db
API_GETCOLLECTION= os.environ.get("API_GETCOLLECTION") # internal API to get collection contents as excel
#API_CHANGECOLLECTION= os.environ.get("API_CHANGECOLLECTION") # API to change collections you are talking to
API_GET_EXISTING_CONV= os.environ.get("API_GET_EXISTING_CONV") # API to get existing conversation ids
API_LOGOUT= os.environ.get("API_LOGOUT") # API to logout (delete session var from redis)
API_GET_AVAILABLE_DOC_NAMES= os.environ.get("API_GET_AVAILABLE_DOC_NAMES")
API_CREATE_EMPTY_COLLECTION= os.environ.get("API_CREATE_EMPTY_COLLECTION")
API_LOG_FEEDBACK= os.environ.get("API_LOG_FEEDBACK")
# API_GET_MEMORY= os.environ.get("API_GET_MEMORY")

# Ingestion configs
MILVUS_URI= os.environ.get("MILVUS_URI")
MILVUS_USER_ROLE= os.environ.get("MILVUS_USER_ROLE")
MILVUS_ROOT_ROLE= os.environ.get("MILVUS_ROOT_ROLE")
TOKEN= os.environ.get("TOKEN")
VECTOR_TOP_K=  int(os.environ.get("VECTOR_TOP_K"))
DOCLING_IMAGE_STORE= parent_dir + os.environ.get("DOCLING_IMAGE_STORE")
DOCLING_HASH_IMAGESTORE= parent_dir + os.environ.get("DOCLING_HASH_IMAGESTORE")
FILES_DB = parent_dir + os.environ.get("FILES_DB")
DOC_NAME_METADATA= os.environ.get("DOC_NAME_METADATA")

# choose llm serving backend
BACKEND= os.getenv("BACKEND")
EMBED_BACKEND_URL= ""
VLLM_RERANK_URL= os.environ.get("VLLM_RERANK_URL")
VLLM_GEN_URL= os.environ.get("VLLM_GEN_URL")
GEN_CONTEXT_WINDOW= os.environ.get("GEN_CONTEXT_WINDOW")
if BACKEND=="ollama":
  EMBED_BACKEND_URL= os.environ.get("OLLAMA_OPENAI_URL")
elif BACKEND=="vllm":
  EMBED_BACKEND_URL= os.environ.get("VLLM_EMBED_URL")

MODEL_EMBED_SMALL= os.environ.get("MODEL_EMBED_SMALL")
MODEL_EMBED_BIG= os.environ.get("MODEL_EMBED_BIG")
MODEL_RERANK= os.getenv("MODEL_RERANK")

if MODEL_EMBED_BIG not in ["snowflake-arctic-embed2", "nomic-embed-text", "qwen3_embed"]:
  raise HTTPException(status_code=404, detail="Incorrect embedding model config")

# columns in the llm response log excel file
retrieval_observe_columns= ["time","chat_session", "question", "retrievals", "reranked_results", "collection", "LLM_response", "Citations", "Feedback (like/dislike)", "Feedback (Comments)"]

# COLLECTION - EMBEDDING MODEL CONFIG
col_mod= {"support_small":MODEL_EMBED_SMALL, 
          "support":MODEL_EMBED_BIG,
          "sales_small":MODEL_EMBED_SMALL,
          "sales":MODEL_EMBED_BIG,
          "compliance_center": MODEL_EMBED_BIG}

# COLLECTION - DOMAIN CONFIG
collection_type= {"support_small": "WISKI Support", 
          "support": "WISKI Support",
          "sales_small": "Product Information",
          "sales": "Product Information",
          "compliance_center": "Compliance Center"}

# COLLECTION - PROMPT CONFIG
systemprompt= {"support_small": systemprompt_support, 
          "support": systemprompt_support,
          "sales_small": systemprompt_product,
          "sales": systemprompt_product,
          "compliance_center": systemprompt_compliance}

# COLLECTION - CHUNK SIZE CONFIG
mod_chunk = {"support_small": 512,
              "support": 4096,
          "sales_small": 512,
          "sales": 4096, 
          "compliance_center":4096} 

# EMBEDDING MODEL - CHUNKS PER BATCH CONFIG
batch_mod= {MODEL_EMBED_SMALL:64,
            "all-minilm:22m":64,
            MODEL_EMBED_BIG:8
            }

# EMBEDDING MODEL - DIMENSION OF EMBEDDING CONFIG
dim_mod= {"allmini-22m-512":384,
            "all-minilm:22m":384,
            "nomic-embed-text":768,
            "snowflake-arctic-embed2":1024,
            "qwen3_embed": 1024
            }

# EMBEDDING MODELS - NUMBER OF RERANKED CHUNKS CONFIG
topk_mod= {MODEL_EMBED_SMALL:5,
           "all-minilm:22m":5,
           MODEL_EMBED_BIG:3
          }
# EMBEDDING MODEL - PORT NUMBER IN VLLM CONFIG
port_vllm_col= {
  MODEL_EMBED_SMALL:"8080",
  MODEL_EMBED_BIG:"8081"
}

# EMBEDDING MODEL - CONTEXT LENGTH PER MODEL CONFIG
model_len_model= {
 "allmini-22m-512":512,
  "all-minilm:22m":256,
  "nomic-embed-text":8192,
  "snowflake-arctic-embed2":8192, 
  "qwen3_embed":8192
}


# content template
milvus_text_template="{metadata_str} \n{content}"

SECTIONS_TO_REMOVE=["index"]

citation_header= "\n\n" + "## Citations:"

# tokenizer default for sentence splitter (use as a proxy for actual tokenizer to get token length of a string)
dummy_model= 'gpt-3.5-turbo' 

"""Simulated Response of llm"""
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import Response
# temp response: 
response_source= [NodeWithScore(node=TextNode(id_='459157232625644143', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text="Source 1:\nDocument_Name:wiski7_deployment_en.pdf\nSection:19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management\n \n\nWe strongly recommend to use a server certificate signed by an official and/or trusted Certification Authority!\n\nIf you still prefer to use the generated self-signed root CA provided by default, you can stop reading here - which we do not recommend!\n\nIf you want your own certificates to be added to our WISKI Server KeyStore and TrustStore, provide the following files to us using the installation/update wizard dialog of the WISKI7 Server Manager:\n\n- § X.509 Root Certification Authority Certificate File format: PEM\n- If  and  only  if  the  server  certificate  has  a  X.509  Extended  Key  Usage  extension:  the  purposes  TLS  Web  Server\n- § X.509 Server Certificate - including the whole certificate chain up to the root CA File format: PEM Authentication and TLS Web Client Authentication (for RMI) must both be set!\n- § Server Certificate Private Key File format: PEM Private key format: PKCS #8 or PKCS #1 Private key algorithm: EC or RSA\n\n\n\nAlternatively, you can also provide the KeyStore and TrustStore files by yourself. The stores must be in PKCS #12 format and must be password-protected.\n\nWhat is a self-signed certificate?\n\nA self-signed server certificate is typically signed by itself (aka the developer or company, anyone) or - as implemented in WISKI Server - by an untrusted root certification authority which is  generated  by  the  WISKI  Server  setup  (Common  Name Generated Self-Signed Root CA for WISKI Server <host> ).\n\nSome customers as well as KISTERS SysAdmins make also usage of self-signed certificates for internal usage. This is cheaper and more flexible than using certificates signed by an official certification authority. Typically the generated root certification authority  used  to  sign  such  server  certificates  is  then  being  rolled  out  automatically  to  all  machines'  operating  system TrustStore in the network.\n", mimetype='text/plain', start_char_idx=None, end_char_idx=None, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}'), score=0.013571249094776937), NodeWithScore(node=TextNode(id_='459157232625643827', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text='Source 2:\nDocument_Name:wiski7_deployment_en.pdf\nSection:WISKI7 Administration and Configuration\n \n\nThe software programs described in this document and the information contained in this document are confidential and proprietary products of KISTERS or its licensors. KISTERS waives copyright for licensed software users to print out parts of the documentation in hard copy for their own use only.\n\nThis documentation may not be transferred, disclosed, or otherwise provided to third parties. In duplicating any part of this document, the recipient agrees to make every reasonable effort to prevent the unauthorized use and distribution of the proprietary information.\n\nNo parts of this work may be reproduced in any form or by any means - graphic, electronic, or mechanical, including photocopying, recording, taping, or information storage and retrieval systems - without the written permission of the publisher.\n\nKISTERS reserves the right to make changes in specifications and other information contained in this publication without prior notice.\n\nKISTERS makes no warranty of any kind with regard to this material including, but not limited to, the implied warranties or merchantability and fitness for a particular purpose.\n\nKISTERS shall not be liable for any incidental, indirect, special or consequential damages whatsoever (including but not limited to lost profits) arising out of or related to this documentation, the information contained in it or from the use of programs and source code that may accompany it, even if KISTERS has been advised of the possibility of such damages.\n\nAny errors found in any KISTERS  product should be reported to KISTERS where every effort will be made to quickly resolve the problem.\n\nProducts that are referred to in this document may be either trademarks and/or registered trademarks of the respective owners. The publisher and the author make no claim to these trademarks.\n\nAuthor: KISTERS Date of current print: 12/03/2025 Current software version: 7.4.15\n', mimetype='text/plain', start_char_idx=None, end_char_idx=None, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}'), score=0.008136560503352323), NodeWithScore(node=TextNode(id_='459157232625643813', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text='Source 3:\nDocument_Name:WISKI7_Pre-UpdateChecks_7.4.13 SR12.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR13.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR14.pdf\nSection:WISKI 7.4.13 SR12\n \n\nThe software programs described in this document and the information contained in this document are confidential and proprietary products of KISTERS or its licensors. KISTERS waives copyright for licensed software users to print out parts of the documentation in hard copy for their own use only. This documentation may not be transferred, disclosed, or otherwise provided to third parties.\n\nIn duplicating any part of this document, the recipient agrees to make every reasonable effort to prevent the unauthorized use and distribution of the proprietary information.\n\nNo parts of this work may be reproduced in any form or by any means - graphic, electronic, or mechanical, including photocopying, recording, taping, or information storage and retrieval systems - without the written permission of the publisher.\n\nKISTERS reserves the right to make changes in specifications and other information contained in this publication without prior notice.\n\nKISTERS makes no warranty of any kind with regard to this material including, but not limited to, the implied warranties or merchantability and fitness for a particular purpose.\n\nKISTERS shall not be liable for any incidental, indirect, special or consequential damages whatsoever (including but not limited to lost profits) arising out of or related to this documentation, the information contained in it or from the use of programs and source code that may accompany it, even if KISTERS has been advised of the possibility of such damages.\n\nAny errors found in any KISTERS  product should be reported to KISTERS where every effort will be made to quickly resolve the problem.\n\nProducts that are referred to in this document may be either trademarks and/or registered trademarks of the respective owners. The publisher and the author make no claim to these trademarks.\n', mimetype='text/plain', start_char_idx=None, end_char_idx=None, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}'), score=0.00596013630162426), NodeWithScore(node=TextNode(id_='459157232625644144', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text="Source 4:\nDocument_Name:wiski7_deployment_en.pdf\nSection:19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management\n \n\nThis is not the case when using self-signed certificates signed by a generated certification authority as used by default by the WISKI Server.\n\nAn official server certificate is being signed by an official root certification authority, e.g. GlobalSign, which is globally trusted by all operating systems TrustStores and which typically takes money for that.\n\nWhat's a certificate chain?\n\nA certificate  chain  is  an  ordered  list  of  certificates  stored  in  one  file  starting  with  the  server  certificate  via  it's  (optional) intermediate certificates to the root CA certificate.\n\nSuch a chain can be build manually by concatening e.g. server.crt and rootCA.crt in the defined order or automatically by using the Security Tool . 145\n\nMost server processes consume a server certificate chain: directly by consuming a chain in e.g. a PEM file like chain.crt or by consuming a KeyStore which contains in addition to the chain also the protected server private key. The last mechanism is being used by WISKI Server.\n\nThe server certificate chain will be presented to each client connecting to a TLS secured port. The server certificate will be used by the client to verify, among other, that the DNS name matches - you need to be sure to talk with the correct secured endpoint and not something else when presenting e.g. a password, the certificate is valid for the required use case and not expired. Intermediate till root CA certificates will be used by the client to ensure the remote endpoint is trusted by validating the presented root CA with it's counterpart in the client TrustStore.\n\nMy system has multiple hostnames. How to handle that?\n\nCertificates  store  the  mandatory  system  FQDN  DNS  name  into  the CommonName field.  Additional  FQDN  names  can  be specified using the Certificate Subject Alternative Name Extension.\n\nThe WISKI Server Security Tool is able to generate self-signed certificates using multiple hostnames.\n", mimetype='text/plain', start_char_idx=None, end_char_idx=None, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}'), score=0.004832546148686673), NodeWithScore(node=TextNode(id_='459157232625644084', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, metadata_template='{key}: {value}', metadata_separator='\n', text="Source 5:\nDocument_Name:wiski7_deployment_en.pdf\nSection:18 Using the WISKI Standard Portal > 18.3 Portal Administration (since WISKI 7.4.11 SR7)\n \n\nFor all later logins the user will see the application he has used when he logged out the last time.</td></tr><tr><td>Application Selector Style</td><td>The application selector is the drop down list which appears when you click on the application title in the title bar (to the right of the KISTERS logo).</td></tr><tr><td>Logo Title</td><td>The logo title is the fly-over text appearing when you move with your mouse over the KISTERS logo at the top left.</td></tr></tbody></table>\n\nUser Settings\n\nThe Portal Admin application provides the main user management system in the WISKI7 Standard Portal.\n\n- § When you want to create a new user, you need to create and configure them in WISKI7 first. Once a user has logged in at the portal (needed for registration), he/she will appear in the portal's user list like depicted below.\n- § Double-click on a user entry to manage the main user settings. Note :  The  WISKI7  Standard  Portal  provides  all  major  applications  as  roles.  So  you  can  easily  maintain  which  user  has access to which application.\n- § Click the [ SAVE ] button in the lower right corner to save your changes.\n", mimetype='text/plain', start_char_idx=None, end_char_idx=None, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}'), score=0.004103538561619608)]
response_text= "This is this according to Source 3 and Source 1"

response_artificial= Response(response= response_text, source_nodes= response_source)


""" Simulated reranked results"""

reranked_articial= [{'Doc_id': '459157232625644143', 'Document': "Document_Name:wiski7_deployment_en.pdf\nSection:19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management\n \n\nWe strongly recommend to use a server certificate signed by an official and/or trusted Certification Authority!\n\nIf you still prefer to use the generated self-signed root CA provided by default, you can stop reading here - which we do not recommend!\n\nIf you want your own certificates to be added to our WISKI Server KeyStore and TrustStore, provide the following files to us using the installation/update wizard dialog of the WISKI7 Server Manager:\n\n- § X.509 Root Certification Authority Certificate File format: PEM\n- If  and  only  if  the  server  certificate  has  a  X.509  Extended  Key  Usage  extension:  the  purposes  TLS  Web  Server\n- § X.509 Server Certificate - including the whole certificate chain up to the root CA File format: PEM Authentication and TLS Web Client Authentication (for RMI) must both be set!\n- § Server Certificate Private Key File format: PEM Private key format: PKCS #8 or PKCS #1 Private key algorithm: EC or RSA\n\n\n\nAlternatively, you can also provide the KeyStore and TrustStore files by yourself. The stores must be in PKCS #12 format and must be password-protected.\n\nWhat is a self-signed certificate?\n\nA self-signed server certificate is typically signed by itself (aka the developer or company, anyone) or - as implemented in WISKI Server - by an untrusted root certification authority which is  generated  by  the  WISKI  Server  setup  (Common  Name Generated Self-Signed Root CA for WISKI Server <host> ).\n\nSome customers as well as KISTERS SysAdmins make also usage of self-signed certificates for internal usage. This is cheaper and more flexible than using certificates signed by an official certification authority. Typically the generated root certification authority  used  to  sign  such  server  certificates  is  then  being  rolled  out  automatically  to  all  machines'  operating  system TrustStore in the network.", 'Metadata': "{'Date': '12/03/2025', 'Document_Name': 'wiski7_deployment_en.pdf', 'Ingested/Printed': 'Printed date', 'Section': '19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management', 'binary_hash': '15772178506683168712', 'page_no': [[219, 220, 221, 222, 223]]}", 'score': 0.013571249094776937}, {'Doc_id': '459157232625643827', 'Document': 'Document_Name:wiski7_deployment_en.pdf\nSection:WISKI7 Administration and Configuration\n \n\nThe software programs described in this document and the information contained in this document are confidential and proprietary products of KISTERS or its licensors. KISTERS waives copyright for licensed software users to print out parts of the documentation in hard copy for their own use only.\n\nThis documentation may not be transferred, disclosed, or otherwise provided to third parties. In duplicating any part of this document, the recipient agrees to make every reasonable effort to prevent the unauthorized use and distribution of the proprietary information.\n\nNo parts of this work may be reproduced in any form or by any means - graphic, electronic, or mechanical, including photocopying, recording, taping, or information storage and retrieval systems - without the written permission of the publisher.\n\nKISTERS reserves the right to make changes in specifications and other information contained in this publication without prior notice.\n\nKISTERS makes no warranty of any kind with regard to this material including, but not limited to, the implied warranties or merchantability and fitness for a particular purpose.\n\nKISTERS shall not be liable for any incidental, indirect, special or consequential damages whatsoever (including but not limited to lost profits) arising out of or related to this documentation, the information contained in it or from the use of programs and source code that may accompany it, even if KISTERS has been advised of the possibility of such damages.\n\nAny errors found in any KISTERS  product should be reported to KISTERS where every effort will be made to quickly resolve the problem.\n\nProducts that are referred to in this document may be either trademarks and/or registered trademarks of the respective owners. The publisher and the author make no claim to these trademarks.\n\nAuthor: KISTERS Date of current print: 12/03/2025 Current software version: 7.4.15', 'Metadata': "{'Date': '12/03/2025', 'Document_Name': 'wiski7_deployment_en.pdf', 'Ingested/Printed': 'Printed date', 'Section': 'WISKI7 Administration and Configuration', 'binary_hash': '15772178506683168712', 'page_no': [[1, 2]]}", 'score': 0.008136560503352323}, {'Doc_id': '459157232625643813', 'Document': 'Document_Name:WISKI7_Pre-UpdateChecks_7.4.13 SR12.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR13.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR14.pdf\nSection:WISKI 7.4.13 SR12\n \n\nThe software programs described in this document and the information contained in this document are confidential and proprietary products of KISTERS or its licensors. KISTERS waives copyright for licensed software users to print out parts of the documentation in hard copy for their own use only. This documentation may not be transferred, disclosed, or otherwise provided to third parties.\n\nIn duplicating any part of this document, the recipient agrees to make every reasonable effort to prevent the unauthorized use and distribution of the proprietary information.\n\nNo parts of this work may be reproduced in any form or by any means - graphic, electronic, or mechanical, including photocopying, recording, taping, or information storage and retrieval systems - without the written permission of the publisher.\n\nKISTERS reserves the right to make changes in specifications and other information contained in this publication without prior notice.\n\nKISTERS makes no warranty of any kind with regard to this material including, but not limited to, the implied warranties or merchantability and fitness for a particular purpose.\n\nKISTERS shall not be liable for any incidental, indirect, special or consequential damages whatsoever (including but not limited to lost profits) arising out of or related to this documentation, the information contained in it or from the use of programs and source code that may accompany it, even if KISTERS has been advised of the possibility of such damages.\n\nAny errors found in any KISTERS  product should be reported to KISTERS where every effort will be made to quickly resolve the problem.\n\nProducts that are referred to in this document may be either trademarks and/or registered trademarks of the respective owners. The publisher and the author make no claim to these trademarks.', 'Metadata': "{'Date': '27/02/2025 AND 20/05/2025 AND 20/05/2025', 'Document_Name': 'WISKI7_Pre-UpdateChecks_7.4.13 SR12.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR13.pdf AND WISKI7_Pre-UpdateChecks_7.4.13 SR14.pdf', 'Ingested/Printed': 'Printed date AND Printed date AND Printed date', 'Section': 'WISKI 7.4.13 SR12', 'binary_hash': '12700479060414532744 AND 452697434243307737 AND 13783038323381342471', 'page_no': [[1, 2], [1, 2], [1, 2]]}", 'score': 0.00596013630162426}, {'Doc_id': '459157232625644144', 'Document': "Document_Name:wiski7_deployment_en.pdf\nSection:19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management\n \n\nThis is not the case when using self-signed certificates signed by a generated certification authority as used by default by the WISKI Server.\n\nAn official server certificate is being signed by an official root certification authority, e.g. GlobalSign, which is globally trusted by all operating systems TrustStores and which typically takes money for that.\n\nWhat's a certificate chain?\n\nA certificate  chain  is  an  ordered  list  of  certificates  stored  in  one  file  starting  with  the  server  certificate  via  it's  (optional) intermediate certificates to the root CA certificate.\n\nSuch a chain can be build manually by concatening e.g. server.crt and rootCA.crt in the defined order or automatically by using the Security Tool . 145\n\nMost server processes consume a server certificate chain: directly by consuming a chain in e.g. a PEM file like chain.crt or by consuming a KeyStore which contains in addition to the chain also the protected server private key. The last mechanism is being used by WISKI Server.\n\nThe server certificate chain will be presented to each client connecting to a TLS secured port. The server certificate will be used by the client to verify, among other, that the DNS name matches - you need to be sure to talk with the correct secured endpoint and not something else when presenting e.g. a password, the certificate is valid for the required use case and not expired. Intermediate till root CA certificates will be used by the client to ensure the remote endpoint is trusted by validating the presented root CA with it's counterpart in the client TrustStore.\n\nMy system has multiple hostnames. How to handle that?\n\nCertificates  store  the  mandatory  system  FQDN  DNS  name  into  the CommonName field.  Additional  FQDN  names  can  be specified using the Certificate Subject Alternative Name Extension.\n\nThe WISKI Server Security Tool is able to generate self-signed certificates using multiple hostnames.", 'Metadata': "{'Date': '12/03/2025', 'Document_Name': 'wiski7_deployment_en.pdf', 'Ingested/Printed': 'Printed date', 'Section': '19 Appendix > 19.7 FAQ > 19.7.1 FAQ on TLS and Certificate Management', 'binary_hash': '15772178506683168712', 'page_no': [[219, 220, 221, 222, 223]]}", 'score': 0.004832546148686673}, {'Doc_id': '459157232625644084', 'Document': "Document_Name:wiski7_deployment_en.pdf\nSection:18 Using the WISKI Standard Portal > 18.3 Portal Administration (since WISKI 7.4.11 SR7)\n \n\nFor all later logins the user will see the application he has used when he logged out the last time.</td></tr><tr><td>Application Selector Style</td><td>The application selector is the drop down list which appears when you click on the application title in the title bar (to the right of the KISTERS logo).</td></tr><tr><td>Logo Title</td><td>The logo title is the fly-over text appearing when you move with your mouse over the KISTERS logo at the top left.</td></tr></tbody></table>\n\nUser Settings\n\nThe Portal Admin application provides the main user management system in the WISKI7 Standard Portal.\n\n- § When you want to create a new user, you need to create and configure them in WISKI7 first. Once a user has logged in at the portal (needed for registration), he/she will appear in the portal's user list like depicted below.\n- § Double-click on a user entry to manage the main user settings. Note :  The  WISKI7  Standard  Portal  provides  all  major  applications  as  roles.  So  you  can  easily  maintain  which  user  has access to which application.\n- § Click the [ SAVE ] button in the lower right corner to save your changes.", 'Metadata': "{'Date': '12/03/2025', 'Document_Name': 'wiski7_deployment_en.pdf', 'Ingested/Printed': 'Printed date', 'Section': '18 Using the WISKI Standard Portal > 18.3 Portal Administration (since WISKI 7.4.11 SR7)', 'binary_hash': '15772178506683168712', 'page_no': [[176, 177]]}", 'score': 0.004103538561619608}]