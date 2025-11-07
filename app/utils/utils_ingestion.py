from app.Ingestion_workflows.docling_parse_process import Docling_parser
from app.Ingestion_workflows.milvus_ingest import ingest2milvus
from app.config import MILVUS_URI, TOKEN, BACKEND_FASTAPI_LOG, col_mod, DOC_NAME_METADATA, MILVUS_ROOT_ROLE
from app.utils.utils_logging import initialize_logging, logger
from pymilvus import connections, Collection, AsyncMilvusClient, MilvusClient
from pymilvus.exceptions import MilvusException
import os
import asyncio
lock = asyncio.Lock()
# logging config
initialize_logging(BACKEND_FASTAPI_LOG)

class FileUploadValidator:
    def __init__(self, max_size_mb=50):
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    @staticmethod
    def convert_size(size_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    def validate_file(self, file_path):
        try:
            size = os.path.getsize(file_path)
            if size > self.max_size_bytes:
                return False, f"File too large. Maximum size: {FileUploadValidator.convert_size(self.max_size_bytes)}. Please chunk and reupload."
            if file_path.split(".")[-1]!="pdf":
                return False, "Only PDF file format is accepted"
            return True, "File size acceptable"
        except OSError as e:
            return False, f"Error checking file size: {e}"

# def file_exists(file: str, user: str):
#     _, filename= os.path.split(file)
#     full_file=f"./user_data/{user}/saved_parsed_files/docling_{filename}.pickle"
#     if os.path.exists(full_file):
#         return True, "Document parsed and saved"
#     else:
#         return False, "Error while processing, document was not saved"


async def milvus_db_as_excel(collection_name: str):
    logger.info(" Internal API called to retrieve collection %s",collection_name)
    try:
        connections.connect(
                        alias="default",
                        uri=MILVUS_URI,
                        token=TOKEN,
                        )
        collection = Collection(collection_name)
        res = collection.query(expr="id >=0", output_fields=["id","metadata", "text_concat", "text"])
        connections.disconnect("default")
    except Exception as e:
        logger.error(str(e),exc_info=True)
    return res

async def ingest(parsed_doc, file: str, user_name: str, ingest_collection: str, user_milvus_pass: str, conv_id: str):
    """ Postprocess docling parsed data and ingest it to milvus

    Args:
        filename (str): filename which is parsed
        user_name (str): user_name
        ingest_collection (str): collection to ingest to
        user_milvus_pass (str): user milvus password
        conv_id (str): session_id
    """
    # chunk and ingest to milvus
    logger.info("Chunking document of %s elements post parsing for collection %s for session: %s", len(parsed_doc), ingest_collection, conv_id)
    chunked_parsed= await asyncio.to_thread(ingest2milvus.chunk, parsed_doc, collection_name=ingest_collection)
    milvus_ingestor= ingest2milvus()
    data= await milvus_ingestor.preprocess_chunks(dataset=chunked_parsed)
    logger.info("ingesting %s rows for session: %s to collection %s", len(chunked_parsed), conv_id, ingest_collection)
    response, message= await milvus_ingestor.milvus_ingest(uri= MILVUS_URI, token= user_milvus_pass, data= data, collection_name=ingest_collection)
    logger.info(" Milvus reponse to ingestion of %s for session: %s is %s", file, conv_id, message)
    return response, message

async def check_admin(user: str):
    """ Check if user has admin privileges or not
    """
    client= MilvusClient(uri=MILVUS_URI, token= TOKEN)
    if MILVUS_ROOT_ROLE in client.describe_user(user_name=user)["roles"]:
        admin_access_flag= True
    else: 
        admin_access_flag= False
    client.close()
    return admin_access_flag

async def combine_per_doc_metadata(read_collection, doc_list, date_list):
    """ For each chunk, takes the list of docs, date and datesource and creates a list of dict: {doc_name: (date)} 
    Args:
        read_collection (str): collection name
        doc_list (list): List of document names.
        date_list (list): Corresponding list of dates.
    """
    if not (len(doc_list) == len(date_list)):
        logger.error("For %s, date %s, doc list %s for a chunk are not of the same length", read_collection, date_list, doc_list)
        raise ValueError("All input lists must be of the same length.")
    return [{doc: date} for doc, date in zip(doc_list, date_list)]

async def remove_duplicates_by_key(dict_list):
    """
    Deduplicates a list of single-key dictionaries based on keys,
    keeping the first occurrence, and returns a list sorted by keys.
    """
    deduped = {}
    try:
        for d in dict_list:
            key = next(iter(d))
            if key not in deduped:
                deduped[key] = d[key]
    except Exception as e:
        logger.error("error occured while getting metadata to display in the document tab of frontend: %s" , str(e))
    return [{k: deduped[k]} for k in sorted(deduped)]

async def get_doc_in_collection(read_collection: str, uri= MILVUS_URI, token=TOKEN):
    async with lock:
        async_client= AsyncMilvusClient(uri, token= token)
        try:
            existing_data= await async_client.query(collection_name=read_collection, filter="id >=0", output_fields=["id","metadata"])
        except MilvusException as e: 
            logger.warning(str(e))
            raise MilvusException(message=str(e))
        doc_list=[]
        for chunk in existing_data:
            metadata_doc= chunk["metadata"][DOC_NAME_METADATA]
            metadata_list_doc=[i.strip() for i in metadata_doc.split("AND")]
            metadata_date= chunk["metadata"]["Date"]
            metadata_list_date=[i.strip() for i in metadata_date.split("AND")]
            metadata_list= await combine_per_doc_metadata(read_collection=read_collection, doc_list=metadata_list_doc, date_list=metadata_list_date)
            doc_list.extend(metadata_list)
        await async_client.close()
        final_doc_list= await remove_duplicates_by_key(doc_list)
        return final_doc_list

# Usage
if __name__ == "__main__":
    validator = FileUploadValidator(max_size_mb=500)
    a = validator.validate_file("./data/wiskienguide.pdf")
    print(a[1])