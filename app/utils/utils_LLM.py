from app.Ingestion_workflows.docling_parse_process import Docling_parser
from app.Ingestion_workflows.milvus_ingest import ingest2milvus
from app.config import RETRIEVAL_LOG_PATH, BACKEND_FASTAPI_LOG, BACKEND, VLLM_RERANK_URL, MODEL_RERANK, DOCLING_HASH_IMAGESTORE, FASTAPI_URL, retrieval_observe_columns, citation_header, reranked_articial
from app.utils.utils_logging import initialize_logging, logger
from app.utils.utils_req_templates import RerankResult
from app.utils.utils_LLM_process_inputs import qwen_rerank_preprocess
from app.prompt_config import RETRIEVE_INSTRUCTION
from datetime import datetime
from llama_index.core import Response
from pymilvus import AnnSearchRequest, RRFRanker , AsyncMilvusClient
from pymilvus.model.reranker import BGERerankFunction # type: ignore
from typing import List
import ast
import asyncio
import copy
import httpx
import os
import pandas as pd
import re
# logging config
initialize_logging(BACKEND_FASTAPI_LOG)
lock = asyncio.Lock()

async def thread_safe_deep_copy(lst):
    """Avoid corrupting data incase of multiconcurrency
    """
    async with lock:
        return copy.deepcopy(lst)

async def log_retrievals(retrievals,question: str, user: str, session_id: str, collection_name: str, LLM_response: str, citations= [], reranked_results= []):
    """ For each retrieval in any session for any user, it logs the retrievals into user data in data folder of the user: chat session > question > retrieval > reranked results > LLM_response
    """
    retrieval_path= RETRIEVAL_LOG_PATH.format(user= user)
    retrieval_string= reranked_string= citation_string= ""
    reranked_results_copy= await thread_safe_deep_copy(reranked_results)
    time= str(datetime.now())
    if citations:
        for citation in citations:
            citation_string+=str(citation-1)+ ", "
        citation_string= citation_string.strip(", ")
    for retrieval in retrievals:
        retrieval_string+=str(retrieval)+ "\n\n"
    for results in reranked_results_copy:
        del results["Document"]
        reranked_string+=str(results)+ "\n\n"
    if os.path.exists(retrieval_path):
        df= pd.read_excel(retrieval_path)
        for col in retrieval_observe_columns:
            if col not in df.columns:
                df[col]=""
        df.loc[len(df)] = [time, session_id, question, retrieval_string, reranked_string, collection_name, LLM_response, citation_string, "N/A", "N/A"]
    else:
        df= pd.DataFrame(columns=retrieval_observe_columns)
        df.loc[len(df)] = [time, session_id, question, retrieval_string, reranked_string, collection_name, LLM_response, citation_string, "N/A", "N/A"]
    df.to_excel(retrieval_path, index=False)

async def get_rerank(url:str, payload: dict):
    results=[]
    async with httpx.AsyncClient() as client:
        responses = await client.post(url, json=payload, timeout=10000)
        if responses.status_code==200:
            responses=responses.json()["results"]
            results = [RerankResult(text=d["document"]["text"], score=d["relevance_score"],index= d["index"]) for d in responses]
        return results

async def rerank_documents(query, documents, model_name, device="cuda", top_k=4):
    """
    Rerank a list of documents based on relevance to the query.

    Parameters:
    - query (str): The input query.
    - documents (list of dict): List of dictionaries, each containing:
        - 'id': Unique identifier
        - 'similarity_score': Initial similarity score
        - 'text': Document text
        - 'metadata': Additional metadata (if any)
    - model_name (str): Name of the reranker model.
    - device (str): Compute device ('cpu' or 'cuda:1').
    - top_k (int): Number of top results to return.

    Returns:
    - list of dict: Reranked documents sorted by relevance.
    """
    # Extract only the texts for reranking
    if not model_name:
        model_name="BAAI/bge-reranker-v2-m3"
    documents= documents[0] # only 1 set of retrievals from vector db currently
    texts = [doc['entity']['text_concat'] for doc in documents]

     # Initialize the reranker
    if BACKEND=="ollama":
        try:
            bge_rf = BGERerankFunction(model_name=model_name, device=device)
            results = await asyncio.to_thread(bge_rf, query=query, documents=texts, top_k=top_k)
        except RuntimeError: 
            # try with cpu
            logger.warning("GPU not found, resorting to CPU")
            bge_rf = BGERerankFunction(model_name=model_name, device="cpu")
            results = await asyncio.to_thread(bge_rf, query=query, documents=texts, top_k=top_k)
    elif BACKEND=="vllm":
        if "qwen3" in model_name.lower(): 
            query, texts= qwen_rerank_preprocess(query, texts)
        try:
            payload={"model": model_name, "query": query, "documents": texts, "top_n":top_k}
            results= await get_rerank(url=VLLM_RERANK_URL, payload=payload)
        except Exception as e:
            logger.error(f"VLLM Backend did not work, try again: {str(e)}")
            results={}
    else:
        logger.error("Wrong backend used, check startup script, should be 'ollama' or 'vllm'")
        raise AssertionError("Wrong backend")
    #Map reranked results back to original documents
    reranked_docs = []
    for result in results:
        for doc in documents:
            if doc['entity']['text_concat'] == result.text:
                reranked_doc = {
                    "id": doc['id'],
                    "similarity_score": result.score,  # Update with new score
                    "text": result.text,
                    "metadata": doc['entity']['metadata']
                }
                reranked_docs.append(reranked_doc)
                break
    #Sort documents by new similarity score (descending)
    return sorted(reranked_docs, key=lambda x: x['similarity_score'], reverse=True)

async def milvus_hybrid_retrieve(uri: str, token: str,question:str, collection_name:str, model, k=5, vector_k= 20):
    """ Hybrid retreival using sparse and dense embeddings 

    Args:
        question (str): question for which to retrieve nodes 
        collection_name (str): collection name to search in
        model: Embedding model
        tokenizer: Corresponding tokenizer
        k (int): number of reranked results. Defaults to 5.
        vector_k: number of retrievals pre reranking. 
    """
    milvus_client = AsyncMilvusClient(uri, token= token)
    # Prepare ANNS field for bm25 search (sparse)
    full_text_search_params = {"metric_type": "BM25", "params": {"drop_ratio_build": 0.1}}
    full_text_search_req = AnnSearchRequest(data=[question], anns_field="sparse_embedding", param=full_text_search_params, limit=vector_k)
    # Prepare ANNS field for semantic search (dense)
    question_dict= {"text_concat": question}
    quetion_embedding= await ingest2milvus.encode_text(question_dict, model=model, instruct=RETRIEVE_INSTRUCTION)
    question_dense_embeddings = [v for v in quetion_embedding["dense_embedding"]]
    dense_search_params = {"metric_type": "COSINE", "params": {"ef": 25}}
    dense_req = AnnSearchRequest(
        data=question_dense_embeddings, anns_field="dense_embedding", param=dense_search_params, limit=vector_k,
    )
    
    # Search topK docs based on dense and sparse vectors and rerank with RRF.
    search_results = await milvus_client.hybrid_search(collection_name=collection_name,
        reqs= [full_text_search_req, dense_req], ranker=RRFRanker(), limit=vector_k, output_fields=["id", "text_concat", "metadata"]
    )
    logger.info("Retrieval completed, reranking now.")
    reranked_results = await rerank_documents(query=question_dict["text_concat"], documents= search_results, device="cuda", top_k=k, model_name=MODEL_RERANK)
    await milvus_client.close()
    # store vector db retrievals as list of dict (only useful for retrieval observability after reranker)
    retrievals_list= []
    for res in search_results:
        for retrieval in res:
            retrieved_doc= {}
            retrieved_doc["Doc_id"]= str(retrieval["id"])
            retrieved_doc["Metadata"]= str(retrieval["entity"]["metadata"])
            retrieved_doc["score"]= retrieval["distance"]
            retrievals_list.append(retrieved_doc)
    # Store reranked results as list of dict
    reranked_list= []
    for result in reranked_results:
        retrieved_doc= {}
        retrieved_doc["Doc_id"]= str(result["id"])
        retrieved_doc["Document"]= str(result["text"])
        retrieved_doc["Metadata"]= str(result["metadata"])
        retrieved_doc["score"]= result["similarity_score"]
        reranked_list.append(retrieved_doc)
    logger.info("Reranking done.")
    return retrievals_list, reranked_list

'''Citation modules'''

def get_page_from_reranked(reranked_list: List, node_id: List):
    """ Get a tuple of chunk info for each node id corresponding to sources mentioned by llm
    """
    doc_lookup = {retrieval["Doc_id"]: retrieval for retrieval in reranked_list}
    
    citation_dict = {}
    for id_pair in node_id:
        source, doc_id = next(iter(id_pair.items()))
        retrieval = doc_lookup.get(doc_id)
        
        if retrieval:
            metadata = ast.literal_eval(retrieval["Metadata"])
            binary_hash = metadata["binary_hash"].split("AND")[0].strip()
            page_no = metadata["page_no"][0]
            section = metadata["Section"]
            document_name = metadata["Document_Name"].split("AND")[0].strip()
            citation_dict[source] = (binary_hash, page_no, section, document_name)
    return citation_dict


def image_to_html(images):
    """
    Returns an HTML link. When clicked, opens a new window with a gallery of the cited PNG images.
    Each image is clickable and opens at full size in a new tab/window.
    """
    # Construct full URLs for JavaScript
    js_images = [f"'{FASTAPI_URL}{url}'" for url in images]
    js_array = "[" + ",".join(js_images) + "]"
    js_code = f"""
        var imgs = {js_array};
        var html = "<div style='display:flex;flex-wrap:wrap;gap:12px;padding:24px;background:#222;'>";
        for (var i=0; i<imgs.length; i++) {{
            html += "<img src='" + imgs[i] + "' style='width:240px;max-width:100%;margin:5px;border:1px solid #ccc;cursor:zoom-in;background:#fff;' onclick=\\"window.open('" + imgs[i] + "', '_blank', 'width=900,height=1200');\\" title='Click to zoom'/>";
        }}
        html += "</div>";
        var w = window.open('', 'gallery', 'width=1200,height=900');
        w.document.write('<html><head><title>Cited Pages</title></head><body style=\\'margin:0;\\'>' + html + '</body></html>');
        w.document.title = 'Cited Pages';
        return false;
    """
    safe_js = js_code.replace('"', '&quot;').replace('\n', ' ')
    link_html = f"<a href='#' onclick=\"{safe_js}\">View cited pages</a>"
    return link_html

def get_image_by_page(citation_tuple):
    """ For each source, get the Docling document and its base64 image by page number range. Embed is html tag and return. 

    Args:
        citation_tuple (tuple):  tuple of file hash, pageno, section, document_name
    """
    image_store= Docling_parser.get_store(DOCLING_HASH_IMAGESTORE)
    file_path= image_store.get(citation_tuple[0])
    imgs= ["/doc_store/"+ file_path.split("/")[-1] + f"/{page_no}.png" for page_no in citation_tuple[1]]
    return image_to_html(images=imgs)

def cite(result: Response, top_k: int, conv_id: str, reranked_list: List):
    """ For each "source" mentioned in llm response, get the match number and match it to the source nodes provided in context. 
        Then get the page, binary hash and document name for all matched nodes and get image by hash and page number embedding them as citation
    Args:
        result (Response object): Response from llm without citation
        top_k (int): Max no reranked chunks
        conv_id (str): session id
        reranked_list (List): list of reranked chunks to get metadata from
    """
    def get_next_words(text, target_word):
        pattern = f"{target_word}\\s+(\\d+)"
        matches = re.finditer(pattern, text)
        return [match.group(1) for match in matches]
    node_id=[]
    if "Source" in result.response:
        match= get_next_words(result.response, "Source")
    else:
        return result.response, []
    match= list(set(match))
    match= [int(x) for x in match]
    match= sorted(match)
    try:
        match= [i for i in match if i <= top_k]
    except Exception as e:
        logger.error("Citation function error for session: %s. Error: %s", conv_id, str(e))
        return result.response, []
    if match:
        logger.info("Creating citations for session: %s", conv_id)
        # if match is found, make citation
        response= result.response + citation_header
        for i in match:
            node_id.append({f"Source {i}":result.source_nodes[i-1].node.node_id})
        citation_dict= get_page_from_reranked(reranked_list=reranked_list, node_id=node_id)
        logger.info("Citation dict for the response: %s", citation_dict)
        for source, citation in citation_dict.items():
            image_by_source= get_image_by_page(citation)
            response= response + "\n\n" + f"**{source}**" + "\nDocument Name: " + citation[3] + "\nSection: " +  citation[2] + "\n" + image_by_source       
        return response, match
    else:
        return result.response, []