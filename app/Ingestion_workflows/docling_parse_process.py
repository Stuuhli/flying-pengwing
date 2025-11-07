from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode
)
from PIL import Image
import sys
sys.path.append("/home/sarvagya/tool_exploration/poc_apptainer/PoCs/")
from app.config import DOC_NAME_METADATA, DOCLING_IMAGE_STORE, DOCLING_HASH_IMAGESTORE, BACKEND_FASTAPI_LOG, SECTIONS_TO_REMOVE
from app.utils.utils_logging import initialize_logging, logger
from docling_core.transforms.serializer.html import HTMLTableSerializer
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from docling_core.types.doc.document import SectionHeaderItem, TableItem
from docling_core.types.doc import DoclingDocument
from io import BytesIO
from llama_index.core import Document
from pathlib import Path
from pypdf import PdfReader
from rapidfuzz import fuzz
from rapidfuzz import process
from typing import Literal, List, Union
import base64
import hashlib
import html
import io
import json
import os
import pickle
import re
initialize_logging(BACKEND_FASTAPI_LOG)

script_dir = os.path.abspath(os.getcwd())
#parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(script_dir)

class Docling_Process():
    """ Postprocessing of parsed file by fixing title format and merge nodes part of same section
    """
    def __init__(self, doc_parsed, filename: str):
        self.doc_parsed= doc_parsed
        self.file_name= filename
        pass

    @staticmethod
    def has_numbers(inputString: str):
        """ Check if section numebr is present

        Args:
            inputString (str): title of the section

        Returns:
            Bool: True if title has digits as section number
        """
        if bool(re.search(r'\s\d+$', inputString)):
            if not bool(re.search(r"^\d+(\.\d){0,4}\b", inputString)):
                return True
        return False
    
    @staticmethod
    def switch_number(inputString: str):
        """ Fix title by appending title number at the start of the title

        Args:
            inputString (str): title

        Returns:
            processed_string: fixed title with number at start
        """
        m = re.search(r'\s\d+$', inputString)
        if m:
            digit= m.group()
            processed_string= inputString.replace(digit, "").strip()
            proc_list=processed_string.split()
            proc_list.insert(0, digit.strip())
            processed_string= " ".join(proc_list)
            return processed_string
    
    @staticmethod
    def fix_title_format(inp_str: str):
        """ Main function which fixes title if it has numbers and returns it

        Args:
            inp_str (str): title of the node

        Returns:
           fixed title
        """
        if Docling_Process.has_numbers(inp_str):
            out_str= Docling_Process.switch_number(inp_str)
            return out_str
        else:
            return inp_str
    
    def get_page_range(self, titles: List):
        """ get the page range for each section header item and then remove any duplicate page for each header item

        Returns:
            orig_header: dict with section header as key and page range as value
        """
        prev_section_header=""
        orig_head= {}
        count_header=0
        count_table=0
        for span in self.doc_parsed.spans:
            if isinstance(span.item, SectionHeaderItem):
                count_header+=1
                prev_section_header= span.item.orig
                if span.item.orig not in orig_head:
                    orig_head[span.item.orig]= [[span.item.prov[0].page_no]]
                else: 
                    orig_head[span.item.orig].extend([[span.item.prov[0].page_no]])
            elif prev_section_header:
                orig_head[prev_section_header][-1].append(span.item.prov[0].page_no)
            elif isinstance(span.item, TableItem): 
                count_table+=1
        # if no header is sectionheaderitem and the entire document is a table
        if count_header==0 and count_table==len(self.doc_parsed.spans): 
            for title in titles: 
                title_header=  next(iter(title))
                match= re.search(r"<tr[^>]*>\s*(?:<th\b[^>]*>.*?</th>\s*)+\s*</tr>",title_header, re.I)
                header_row_html = match.group(0) if match else None
                title[header_row_html]= title.pop(title_header)
                orig_head[header_row_html]= [[span.item.prov[0].page_no for span in self.doc_parsed.spans]]
        elif count_header==0: 
            for title in titles:
                orig_head[next(iter(title))]= [[span.item.prov[0].page_no for span in self.doc_parsed.spans]]
        # remove duplicate
        for header in orig_head.keys():
            for index,page_set in enumerate(orig_head[header]):
                orig_head[header][index]=sorted(list(set(page_set)))
        return orig_head, titles

    @staticmethod
    def fuzzy_match_lists(list1: List, list2: List, threshold: int=80):
        """For each item in list1, find the best match in list2 using fuzzy matching.
            Only return matches where the similarity score is at least `threshold`.

        Returns:
            matches: List of tuples (item_from_list1, best_match_in_list2, score)
        """
        matches = []
        for item in list1:
            # Using token_sort_ratio handles cases where word order varies.
            match = process.extractOne(item, list2, scorer=fuzz.token_sort_ratio)
            if match is not None:
                best_match, score = match[0], match[1]
                if score >= threshold:
                    matches.append(best_match)
        return matches

    def get_valid_titles(self, doc_list: List, content_table: str, all_titles: List, metadata_titles: List):
        """ Gets the valid titles by first getting all titles which match the regex pattern from headings of nodes. 
            Then it checks the remaining titles and compares them to cleaned text in content table. If there are any common titles it gets appended to titles from regex matching.  

        Args:
            doc_list (List(Document)): List of all nodes (with non significant titles as well)
            content_table (str): text from table of contents
            all_titles (List(str)): all titles from lists 

        Returns:
           metadata_titles (List(str)): List of all valid titles  
        """
        titles= []
        cleaned_content=[]
        pattern= r"^(?:\d+(?:\.\d+){0,4})(?!\s*[);])(?:\s+.+)?$|^PUC-.+"
        for doc in doc_list: 
            if re.search(pattern, doc.metadata["headings"]):
                titles.append(doc.metadata["headings"])
        try: 
            index= all_titles.index(titles[-1])
            rest_titles= all_titles[index + 1:]
        except Exception as e:
            logger.error(str(e))
            rest_titles=[]
            pass
        
        if content_table:
            content_list= content_table.split("\n")[1:]
            content_list= [content.replace("</td><td>", " ") for content in content_list]
            content_list= [re.sub(r'</?(table|tr|td|th|thead|tbody)[^>]*>', '% HTML TAG %', content, flags=re.IGNORECASE) for content in content_list if content]
            for content in content_list: 
                cleaned_content.extend(content.split('% HTML TAG %'))
            cleaned_content = [re.sub(r'[^a-zA-Z\s]', "", content).strip() for content in cleaned_content if content]
            
            common_titles= Docling_Process.fuzzy_match_lists(cleaned_content, rest_titles, threshold=80)
        else:
            common_titles= []
        metadata_titles.extend(titles)
        metadata_titles.extend(common_titles)
        return metadata_titles

    def get_last_index(self, start_index: int, doc_list: List, title_list: List):
        """ Gets index range for list of documents from section header to end of section (before next section or subsection starts) 

        Args:
            start_index (int): index of node with a section/subsection
            doc_list (List(Document)): List of all nodes
            title_list (List(str)): List of all valid titles   

        Returns:
           last_index (int): last index of node before subsequent section/subsection
        """
        last_index= start_index
        if start_index!=len(doc_list)-1: 
            for next_index, next_doc in enumerate(doc_list[start_index+1:]):
                last_index= next_index + start_index
                if next_doc.metadata["headings"] in title_list:
                    break
                if last_index== len(doc_list)-2:
                    if doc_list[last_index+1] not in title_list:
                        last_index+=1
        return last_index
    
    @staticmethod
    def remove_unwanted_text(text: str):
        """ Function to remove any unwanted text to not confuse the LLM
        """
        unwanted_list= ["(see screenshot below)", "<!-- image -->"]
        for phrase in unwanted_list:
            text= text.replace(phrase, "").strip()
        return text
    
    def copy_node_properties(self, doc_list: List, start_index: int, last_index: int):
        """ Creates new node based on List of Document properties where
            a. nodeids are concatenated
            b. section metadata of the first node (the node at start index)
            c. page range from start index to last index +1 (to cater cases where subsection flows into next page)
            d. text is concatenated and titles are removed

        Args:
            doc_list (List[Document]): List of Documents (processed)
            start_index (int): Index of parent node
            last_index (int): Index of farthest child

        Returns:
            node: new node
        """

        node_id= ("&").join([doc.id_ for doc in doc_list[start_index:last_index+1]])

        # get section and page metadata and make a dict
        section_metadata= doc_list[start_index].metadata['headings']
        start_page= (doc_list[start_index].metadata['page_no'])
        end_page= (doc_list[last_index].metadata['page_no'])
        page_metadata=[i for i in range(start_page[0], end_page[-1]+1)]
        metadata= {"Section": section_metadata, "page_no": [page_metadata], DOC_NAME_METADATA: self.file_name}

        metadata_template= doc_list[start_index].metadata_template
        text= ("\n\n").join([doc.text.strip() for doc in doc_list[start_index:last_index+1]])
        text= text.replace(metadata ["Section"], "") # remove title since it is in metadata
        node= Document(id_= node_id, metadata= metadata, metadata_template=metadata_template, text=text)
        return node

    def merge_nodes(self, doc_list: List):
        """ Merge nodes by first extracting table of contents and then getting valid titles. 
        Then it gets last index before subsequent section/subsection for each section and merges them. 

        Args:
            doc_list (List[Document]): List of Documents where merge is to take place

        Returns:
            processed_doc (List[Document]): List of Documents with merged nodes as per identified section/subsections
        """
        all_titles=[]
        metadata_title= []
        for doc in doc_list: 
            metadata_title.append(doc.metadata["headings"])
            if "table of contents" in doc.metadata["headings"].lower():
                content_table= doc.text
                break
            else: 
                content_table=""
        for doc in doc_list: 
            all_titles.append(doc.metadata["headings"])
        titles= self.get_valid_titles(doc_list, content_table, all_titles, metadata_title)
        processed_doc=[]
        for index, docs in enumerate(doc_list):
            if docs.metadata["headings"] in titles: 
                last_index= self.get_last_index(index, doc_list, titles) # get last index of the node in the same section or subsection
                node= self.copy_node_properties(doc_list=doc_list, start_index=index, last_index=last_index) # create new node preserving the desired properties specific to that node and concatenating text of same section
                processed_doc.append(node)
        return processed_doc

    def process_doc(self):
        """ Post processing orchestration function: 
        1. Fix the title format of the nodes to the desired: {title number} {title text} 
        2. Alter metadata to have the Document name, section position, page no as 3 elements
        3. Merge nodes into one to form a heirarchical structure where each node corresponds to a section/subsection and its text, at the same time preserving the relationships between nodes


        Returns:
            final_processed_doc: Processed list of documents
        """
        preprocessed_doc_list=[]
        self.doc_parsed.text= Docling_Process.remove_unwanted_text(self.doc_parsed.text)
        texts= self.doc_parsed.text.split("\n## ")
        # text processing to remove html codes and \\_
        texts= [html.unescape(i) for i in texts]
        texts= [re.sub(r'\\_', '_', i)  for i in texts]
        titles= [{(re.sub(r"^## ", "", i.split("\n")[0], flags=re.MULTILINE)).strip():i} for i in texts]
        orig_head, titles= self.get_page_range(titles)
        for ind,text in enumerate(texts):
            preprocessed_doc= Document(text=text)
            orig_heading= list(titles[ind].keys())[0]
            preprocessed_doc.metadata["Orig_headings"]= orig_heading
            preprocessed_doc.metadata["headings"]=  Docling_Process.fix_title_format(orig_heading)
            try:
                preprocessed_doc.metadata["page_no"]= orig_head[orig_heading][0]
                orig_head[orig_heading]= orig_head[orig_heading][1:]
            except KeyError:
                preprocessed_doc.metadata["page_no"]=[]
            preprocessed_doc_list.append(preprocessed_doc) 
        if not preprocessed_doc_list[0].metadata["page_no"]:
            preprocessed_doc_list= preprocessed_doc_list[1:]
        processed_doc_list = self.merge_nodes(preprocessed_doc_list)
        return processed_doc_list

class Docling_parser():
    
    @staticmethod
    def save_object(obj, file: str, filetype: Literal["pickle", "json"]= "pickle"):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if file.split(".")[-1] != filetype:
                logger.error("File %s is not a %s although %s option was passed", file, filetype, filetype)
                raise Exception("Invalid file type during file parsing")
        try:
            if filetype=="pickle":
                with open(file, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif filetype=="json":
                with open(file, "w", encoding="utf-8") as f:   
                    json.dump(obj, f)    
        except Exception as ex:
            logger.error("Error during %s dumping object %s (Possibly unsupported): %s", filetype, file, str(ex))
            raise ValueError(str(ex))
    
    @staticmethod
    def get_date(filename:str):
        reader= PdfReader(filename)
        metadata= reader.metadata
        return metadata.creation_date.strftime("%d-%m-%Y")
        
        
    @staticmethod
    def add_metadata_docling(docling_parsed, date: str, binary_hash: str):
        for doc in docling_parsed:
            doc.metadata["Date"]=date
            doc.metadata["binary_hash"]= binary_hash
        return docling_parsed

    @staticmethod
    def parse_docling(doc_converter, source: str):
        doc= doc_converter.convert(source=source).document
        return doc
    
    @staticmethod
    def get_store(path: str):
        """ Creates a store using the path if not present and then returns it as a dict

        Args:
            path (str): path of the store

        Returns:
            imagestore: dict
        """
        # if file not present, create file and return empty dict
        if not os.path.exists(path):
            open(path, 'w').close()
            return {}
        if os.stat(path).st_size == 0:
            with open(path, 'w') as file:
                json.dump({}, file)
        with open(path, 'r') as f:
            imagestore = json.load(f)
        return imagestore
    
    def save_doc_png(self, doc, folder: str, json_path: str):
        """Saves json (of docling doc) and png for each base64 by compressing first and then saving as per page number in folder

        Args:
            doc (Docling_document): docling doc
            folder (str): file where page images for docling doc have to be saved
        """
        os.makedirs(folder, exist_ok=True)
        # save png
        for page in doc.pages.keys():
            uri= str(doc.pages[page].image.uri)
            if uri.startswith("data:image"):
                base64_str = uri.split(",")[1]
            image_data= base64.b64decode(base64_str) 
            image = Image.open(io.BytesIO(image_data))
            output_path = folder + f"/{page}.png"
            image.save(output_path, format="PNG", optimize=True, compress_level=9)
        # save docling doc in same folder
        doc.save_as_json(json_path)
    
    @staticmethod
    def serialize_docling(doc):
        serializer = MarkdownDocSerializer(
           doc=doc, table_serializer=HTMLTableSerializer())
        ser_result = serializer.serialize() 
        return ser_result
    
    def build_hierarchy(self, nodes: List):
        """ Builds a hierarchy of sections adding it to the metadata

        Args:
            nodes (List[TextNode]): Docling parsed doc without sections metadata

        Returns:
            sections: Docling parsed doc with sections metadata
        """
        # Regex to extract the section number and title
        section_pattern = r"^(\d+(\.\d+)*)(?:\s+)?(.+)"
        hierarchy = []
        for node in nodes:
            heading = node.metadata["Section"]
            # Split by newline to separate title from the rest
            match = re.match(section_pattern, heading)
            if match:
                # Extract the section number and title
                section_number = match.group(1).strip()
                section_title = match.group(3).strip()
                # Combine the number and title for the current section
                current_section = f"{section_number} {section_title}"
                # Maintain a hierarchical stack
                while hierarchy and not section_number.startswith(hierarchy[-1].split()[0]):
                    hierarchy.pop()
                hierarchy.append(current_section)
                # Create the full hierarchy path
                current_heirarchy= " > ".join(hierarchy)
                node.metadata["Section"]= current_heirarchy
            else:
                node.metadata["Section"]= heading
        return nodes
    
    @staticmethod
    def remove_sections(processed_doc: List):
        """ Remove certain sections which have no relevant info from the document

        Args:
            processed_doc (List): doc which has been processed and heirarchy has been built

        Returns:
            processed_doc_removed: doc with certain sections removed
        """
        processed_doc_removed= processed_doc.copy()
        for doc in processed_doc:
            if doc.metadata["Section"].lower() in SECTIONS_TO_REMOVE:
                processed_doc_removed.remove(doc)
        return processed_doc_removed

    def post_process(self, doc_serialized, filename: str, date_extracted: str, binary_hash: str):
        try: 
            process_obj= Docling_Process(doc_parsed=doc_serialized, filename= filename)
            processed_doc= process_obj.process_doc()
        except Exception as e: 
            logger.error("Error while post processing the parsed file %s: %s", filename, str(e))
            raise Exception(f"Post processing of parsed file unsuccessful: {str(e)}")
        hierarchical_processed_doc= self.build_hierarchy(processed_doc)
        processed_doc= Docling_parser.add_metadata_docling(docling_parsed=hierarchical_processed_doc, date=date_extracted, binary_hash= binary_hash)
        final_processed_doc= Docling_parser.remove_sections(processed_doc=processed_doc)
        return final_processed_doc

    @staticmethod
    def create_file_hash(path_or_stream: Union[BytesIO, Path]) -> str:
        """Create a stable page_hash of the path_or_stream of a file"""

        block_size = 65536
        hasher = hashlib.sha256(usedforsecurity=False)

        def _hash_buf(binary_stream):
            buf = binary_stream.read(block_size)  # read and page_hash in chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = binary_stream.read(block_size)

        if isinstance(path_or_stream, Path):
            with path_or_stream.open("rb") as afile:
                _hash_buf(afile)
        elif isinstance(path_or_stream, BytesIO):
            _hash_buf(path_or_stream)
        return hasher.hexdigest()

    def docling_ingest(self, file: str, collection_name: str):
        _, filename= os.path.split(file)
        image_store= Docling_parser.get_store(DOCLING_HASH_IMAGESTORE) # get the image store dict which is stored as json

        accelerator_options = AcceleratorOptions(
            num_threads=16, device=AcceleratorDevice.CUDA)
        pipeline_options_v1 = PdfPipelineOptions()
        pipeline_options_v1.accelerator_options = accelerator_options
        pipeline_options_v1.do_ocr = False
        pipeline_options_v1.do_table_structure = True
        pipeline_options_v1.table_structure_options.do_cell_matching = True
        pipeline_options_v1.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options_v1.generate_page_images= True
        pipeline_options_v1.images_scale= 2.0
        custom_accelerated_v1= DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options_v1,
                    )
                }
            )
        # markitdown for date extraction
        try:
            # md = MarkItDown(enable_plugins=False)
            # result = md.convert(file)
            date_extracted= Docling_parser.get_date(filename= file)
            logger.info("Date extracted for markitdown for %s: %s", filename, date_extracted)
        except Exception as e:
            logger.error("Error with date extraction for file %s : %s", file, str(e))
            date_extracted= "N/A"
        # docling to get image and json of parsed data
        file_hash= Docling_parser.create_file_hash(Path(file)) 
        image_folderpath= DOCLING_IMAGE_STORE.format(filename=file_hash) 
        image_folder_jsonpath= image_folderpath + f"/{file_hash}.json"
        if file_hash in image_store.keys() and Path(image_folderpath).exists():
            logger.info("File %s already parsed by docling: using the cached file for downstream ingestion at collection %s", filename, collection_name)
            doc_json= DoclingDocument.load_from_json(image_folder_jsonpath)
        else: 
            logger.info("Cache not found for this file %s. Continuing with parsing.", filename)
            try:
                doc_json= Docling_parser.parse_docling(doc_converter=custom_accelerated_v1, source=file)   
                doc_json.origin.binary_hash= file_hash   
            except Exception as e:
                logger.error("Error with docling parsing for file %s : %s", file, str(e))
                raise Exception(f"Docling parsing failed for file {file}")
            image_store[doc_json.origin.binary_hash]= image_folderpath # store file path for that image using docling hash as key
            logger.info("Saving png for file %s, total pages: %s", filename, len(doc_json.pages))
            self.save_doc_png(doc=doc_json, folder=image_folderpath, json_path= image_folder_jsonpath)
            Docling_parser.save_object(image_store, file=DOCLING_HASH_IMAGESTORE, filetype="json") # save the image store per binary hash as json
            logger.info("Docling parsing successful for file %s", filename)
        doc_json.origin.binary_hash= file_hash
        # serialize the parsed file 
        doc_ser_result= Docling_parser.serialize_docling(doc_json)
        # post process the serialized file and add new metadata
        final_processed_doc= self.post_process(doc_serialized=doc_ser_result, 
                                               filename=filename, date_extracted=date_extracted, 
                                               binary_hash=str(doc_json.origin.binary_hash))
        return final_processed_doc

if __name__ == "__main__":
    parsing_obj= Docling_parser()
    file= "/home/sarvagya/tool_exploration/poc_apptainer/PoCs/data/WISKI7_Pre-UpdateChecks_7.4.13 SR12.pdf"
    parsed_doc= parsing_obj.docling_ingest(file=file, collection_name="test")

