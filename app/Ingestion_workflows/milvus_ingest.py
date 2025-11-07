from app.config import col_mod, batch_mod, dim_mod, port_vllm_col, model_len_model, mod_chunk, milvus_text_template, DOC_NAME_METADATA, BACKEND_FASTAPI_LOG, EMBED_BACKEND_URL, BACKEND, dummy_model
from app.utils.utils_logging import initialize_logging, logger
from datasets import Dataset
from llama_index.core.node_parser import SentenceSplitter
from pymilvus import MilvusClient, AsyncMilvusClient, DataType, Function, FunctionType
from typing import Optional, List
import tiktoken
import asyncio
import httpx
import copy
import re
initialize_logging(BACKEND_FASTAPI_LOG)

class ingest2milvus():
    bm25_function = Function(
        name="text_bm25_emb", # Function name
        input_field_names=["text_concat"], # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse_embedding"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,
    )
    
    @staticmethod
    async def get_embedding(url:str, payload: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10000)
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def encode_text(batch, model, backend_url=EMBED_BACKEND_URL, backend= BACKEND, instruct: str= None):
        if backend=="vllm":
            backend_url= backend_url.format(PORT=port_vllm_col[model])
            if instruct and "qwen3" in model:
                input= f'Instruct: {instruct}\nQuery:{batch["text_concat"]}'
            else:
                input= batch["text_concat"]
            payload={
                "model": model,
                "input": input,
                "truncate_prompt_tokens": model_len_model[model],
                "keep_alive":"59m"
            }
        elif backend=="ollama":
            payload={
                "model": model,
                "input": batch['text_concat'],
                "truncate": True,
                "keep_alive":"59m"
            }
        embeddings = await ingest2milvus.get_embedding(
            url=backend_url,
            payload=payload
            )
        if backend=="vllm":
            batch["dense_embedding"]= [emb["embedding"] for emb in embeddings["data"]]
        elif backend=="ollama":
            batch["dense_embedding"]= [emb for emb in embeddings["embeddings"]]
        return batch

    @staticmethod
    async def process_batch(batch, model):
            logger.info("Encoding text using %s", model)
            return await ingest2milvus.encode_text(batch, model=model)

    async def preprocess_chunks(self, dataset):
        """ Pipeline to process the dataset

        Args:
            dataset: dataset object

        Returns:
            data: dataset object with cleaned columns
        """
        data = dataset.map(remove_columns=['metadata_template', 'metadata_separator', 'text_template'],)
        return data
    
    @staticmethod
    def get_token_len(dummy_model: str, text: str):
        encoding = tiktoken.encoding_for_model(dummy_model)
        return len(encoding.encode(text))

    @staticmethod
    def clean_metadata(parsed_docs: List, dummy_model: str):
        for doc in parsed_docs: 
            if ingest2milvus.get_token_len(dummy_model=dummy_model, text=doc.metadata["Section"])>256: 
                doc.metadata["Section"] =  doc.metadata["Section"].split()[0]
        return parsed_docs

    @staticmethod
    def merge_small(docs: List, chunk_size: int, dummy_model: str):
        """ Do small to big chunking of any chunk which is small

        Args:
            docs (list): list of nodes post chunking and making into a dict
            chunk_size (int): chunk size for the collection
            dummy_model (str): dummy model name

        Returns:
            merged_nodes: merged node names
        """
        merged_nodes = []
        current_node = docs[0].copy()
        numbered_pattern= r"^(?:\d+(?:\.\d+){0,4})(?!\s*[);])(?:\s+.+)?$"  # should be same as the pattern in get_valid_titles
        for i in range(1, len(docs)):
            next_node = docs[i]
            # Calculate the number of tokens if we merge
            combined_text = current_node["text_concat"] + "\n\n" + next_node["text_concat"]
            num_tokens_combined = ingest2milvus.get_token_len(dummy_model=dummy_model, text=combined_text)
            if re.search(pattern=numbered_pattern, string=current_node["metadata"]["Section"].split("AND")[0].strip()):
                # merge for numbered sections nodes
                # TODO: see that merges to same sublevel (9.2.3.4 should stick to 9.2.4 but not 9.3)
                if next_node["metadata"]["Section"].split()[0].strip()==current_node["metadata"]["Section"].split()[0].strip() and num_tokens_combined < int(1/2* chunk_size):
                    current_node["text_concat"] = combined_text
                    current_node["text"] = current_node["text"] + "\n" + next_node["text"]
                    current_node["metadata"]["Section"] = current_node["metadata"]["Section"] + " AND " + next_node["metadata"]["Section"]
                    current_node["metadata"]["page_no"][0].extend(next_node["metadata"]["page_no"][0])
                else: 
                    merged_nodes.append(current_node)
                    # Start a new current_node with the next_node
                    current_node = next_node.copy()
            elif num_tokens_combined < int(1/2* chunk_size):
                # merge for non numbered sections nodes
                # If it fits, merge the next_node into the current_node
                current_node["text_concat"] = combined_text
                current_node["text"] = current_node["text"] + "\n" + next_node["text"]
                current_node["metadata"]["Section"] = current_node["metadata"]["Section"] + " AND " + next_node["metadata"]["Section"]
                current_node["metadata"]["page_no"][0].extend(next_node["metadata"]["page_no"][0])
            else:
                # If it doesn't fit, add the current_node to our list of merged nodes
                merged_nodes.append(current_node)
                # Start a new current_node with the next_node
                current_node = next_node.copy()
        merged_nodes.append(current_node)
        return merged_nodes

    @staticmethod
    def chunk(doc: List, collection_name: str):
        chunk_size=mod_chunk[collection_name]
        if collection_name in list(mod_chunk.keys()):
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=int(0.1*chunk_size))
        else: 
            raise ValueError("Incorrect collection")
        doc= ingest2milvus.clean_metadata(doc.copy(), dummy_model= dummy_model)
        nodes = splitter.get_nodes_from_documents(doc)
        nodes_json=[]
        for node in nodes:
            # add metadata to text in text_concat column
            metadata_str="" 
            for key,value in node.metadata.items():
                if key=="Section":
                    metadata_str= metadata_str + f"{key}:{value}\n"
            nodes_json.append({
                    "text": node.text,
                    "text_concat": milvus_text_template.format(metadata_str= metadata_str,content= node.text),
                    "dense_embedding": [],
                    "metadata": node.metadata,
                    "metadata_template": node.metadata_template,
                    "metadata_separator": node.metadata_separator,
                    "text_template": node.text_template
                })
        # small to large chunking
        merged_nodes_json= ingest2milvus.merge_small(docs=copy.deepcopy(nodes_json), chunk_size=chunk_size, dummy_model=dummy_model)

        # add document name to text_concat to complete the text
        for nodes in merged_nodes_json:
            nodes["text_concat"]= milvus_text_template.format(metadata_str= f"{DOC_NAME_METADATA}:{nodes['metadata'][DOC_NAME_METADATA]}", content= nodes["text_concat"])
            nodes["metadata"]["page_no"][0]= sorted(list(set(nodes["metadata"]["page_no"][0])))
        return Dataset.from_list(merged_nodes_json)
    
    def check_chunks(self, chunk, old_data:list) -> tuple:
        """ Checks for each chunk if its content is present in existing chunks in the collection. 
            Has 2 distinct loops:
            Loop 1: checks if text is same and the if chunk is from the same document.
            Loop 2: For a different document, check if chunk is same, (upsertion with appended metadata will be performed downstream)
        Args:
            chunk : chunk which is being checked for duplicates
            old_data (list): chunks from existing collection
            client (AsyncMilvusClient): to upsert into milvus if same chunk content but differnt document

        Returns:
            1/0: 1 if chunk content is already present in collection, otherweise 0
            list: updated_chunk_batch if same chunk but different document
        """
        exact_text_flag= 0
        for old_chunk in old_data:
            # check if any exact text exists with the name of file-to-be-ingested also present in already inserted data's metadata, then skip entirely
            if chunk["text"]==old_chunk["text"]:
                exact_text_flag=1
                if chunk["metadata"][DOC_NAME_METADATA] in old_chunk["metadata"][DOC_NAME_METADATA]:
                    return 1, {}
        # this loop only gets executed if there are exact matches in texts but document name of the chunk is not present in existing data in collection
        if exact_text_flag==1:
            for old_chunk in old_data:
                # 2 different documents contain the same text: update metadata to contain new document name as well + upsert.
                # remove the document name line at the start from text concat before comparing
                new_text= '\n'.join(chunk["text_concat"].split('\n')[1:])
                old_text= '\n'.join(old_chunk["text_concat"].split('\n')[1:])
                if new_text.lower()==old_text.lower():
                        old_chunk["metadata"][DOC_NAME_METADATA]=old_chunk["metadata"][DOC_NAME_METADATA]+ " AND "+ str(chunk["metadata"][DOC_NAME_METADATA])
                        old_chunk["metadata"]["Date"]=old_chunk["metadata"]["Date"]+ " AND "+ str(chunk["metadata"]["Date"])
                        old_chunk["metadata"]["binary_hash"]=old_chunk["metadata"]["binary_hash"]+ " AND "+ str(chunk["metadata"]["binary_hash"])
                        old_chunk["metadata"]["page_no"].extend(chunk["metadata"]["page_no"])
                        metadata_str=""
                        for key,value in old_chunk["metadata"].items():
                            if key==DOC_NAME_METADATA or key=="header_path":
                                metadata_str= metadata_str + f"{key}:{value}\n"
                        old_chunk["text_concat"]= milvus_text_template.format(metadata_str= metadata_str,content= old_text)
                        return 1, old_chunk
        return 0, {}
    
    async def embed_torch(self, embed_dataset, batch_size: int, model: str):
        dataset_list= Dataset.from_list(embed_dataset)
        dataset_batches = [dataset_list[i : i + batch_size] for i in range(0, len(dataset_list), batch_size)]
        dataset = await asyncio.gather(*(ingest2milvus.process_batch(batch=batch, model= model) for batch in dataset_batches))
        dataset = Dataset.from_dict({key: sum([batch[key] for batch in dataset], []) for key in dataset[0]})
        return dataset.to_list()

    async def deduplicate(self, data: list, old_data: list, client: AsyncMilvusClient, collection_name: str, model: str, batch_size: int):
        """ Removes a chunk from insertion data if its content is already present in the collection (exact match)
        """
        new_data= []
        upsert_chunk_batch=[]
        upsert_dataset=[]
        for chunk in data: 
            flag, upsert_chunk= self.check_chunks(chunk=chunk, old_data=old_data)
            if flag==0:
                # flag is 0 only if the chunk to be inserted is not present in the db. These chunks are carried forward for insertion
                new_data.append(chunk)
            elif flag==1 and not upsert_chunk:
                logger.warning("Duplicate chunks found while inserting to %s", collection_name)
            elif flag==1 and upsert_chunk:
                upsert_chunk_batch.append(upsert_chunk)
        if upsert_chunk_batch:
            logger.warning("Duplicate chunks from different documents found while inserting to %s. Upserting %s chunks with updated chunk/metadata", collection_name, len(upsert_chunk_batch))
            upsert_dataset= await self.embed_torch(embed_dataset=upsert_chunk_batch, batch_size=batch_size, model=model)
            await client.upsert(collection_name=collection_name, data=upsert_dataset)
        if new_data:
            logger.info("Embedding new chunks")
            new_dataset= await self.embed_torch(embed_dataset=new_data, batch_size=batch_size, model=model)
            return new_dataset
        return []

    @staticmethod
    async def create_new_collection(milvus_client: MilvusClient, dim: int, collection_name: str, sparse_function= bm25_function):
        # Create schema if no collection with name found
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, enable_analyzer=True, max_length=60535, default_value="")
        schema.add_field(field_name="text_concat", datatype=DataType.VARCHAR, enable_analyzer=True, max_length=60535)
        schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(sparse_function)

        # Create collection
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=dim,
            schema= schema,
            consistency_level="Strong",  # To enable search with latest data
        )
        index_params = milvus_client.prepare_index_params()

        # Add an index on the dense embedding
        index_params.add_index(
            field_name="dense_embedding",
            metric_type="COSINE",
            index_type="HNSW",
            index_name="vector_index",
            params={"M":98, "efConstruction": 450}
        )
        # Add an index on sparse embedding
        index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX", 
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )
        milvus_client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
        )
        milvus_client.load_collection(collection_name=collection_name)
        logger.info("Created collection %s", collection_name)
        return milvus_client

    async def milvus_ingest(self, uri: str, token: str, data, collection_name: str):
        """ Ingest documents into milvus after creating a schema and indices for dense and sparse embeddings 

        Args:
            uri (str): URL for connecting to milvus standalone
            token (str): login token
            data: dataset to be ingested
            collection_name (str): Collection name to be created. Defaults to COLLECTION_NAME.
            dim (int): Dimension of dense embeddings. Defaults to DIMENSION.
            sparse_function: Sparse embedding funciton to be integrated into schema of milvus collection. Defaults to bm25_function.

        Returns:
            milvus_client: Instantiated milvus client
        """
        model= col_mod[collection_name]
        batch_size= batch_mod[model]
        dim= dim_mod[model]
        data= data.to_list()
        # Instantiate client and insert to collection
        milvus_client = MilvusClient(uri, token= token)
        async_client= AsyncMilvusClient(uri, token= token)
        try: 
            if milvus_client.has_collection(collection_name=collection_name):
                # check if duplicates exist in collection, and filter the duplocates out
                existing_data= await async_client.query(collection_name=collection_name, filter="id >=0", output_fields=["id","metadata", "text", "text_concat"])
                new_data= await self.deduplicate(data=data, old_data= existing_data, client= async_client, collection_name= collection_name, model=model, batch_size=batch_size)
                if not new_data:
                    return False, "File already exists in DB"
                # end of deduplication feature
                await async_client.insert(collection_name=collection_name, data=new_data)
                await async_client.close()
                milvus_client.close()
                return True, "Ingestion Successful"
            milvus_client = await ingest2milvus.create_new_collection(milvus_client=milvus_client, dim= dim, collection_name=collection_name)
            # Embed dataset
            new_data= await self.embed_torch(embed_dataset=data, batch_size=batch_size, model=model)
            # Ingest document into the collection and load it 
            await async_client.insert(collection_name=collection_name, data=new_data)
            await async_client.load_collection(
                collection_name=collection_name
            )
            await async_client.close()
            milvus_client.close()
        except Exception as e: 
            logger.error("Unsuccessful ingestion into %s", collection_name)
            return False, str(e)
        return True, "Ingestion Successful"