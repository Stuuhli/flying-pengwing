from db_observe import get_data
import copy
import json
import os
import pandas as pd
import re
import string
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

def add_ground_truth(log_dataset, eval_path):
    """ Based on question, add ground truth and reference context from log_dataset. 

    Args:
        log_dataset (list[Dict]): json with log retrievals 
        eval_path (str): path of evaluation dataset

    Returns:
        List[Dict]: json with ground truth added
    """
    df_eval= pd.read_excel(eval_path, sheet_name="LLM eval")
    df_eval.reset_index(inplace=True, drop=True)
    for index,question in enumerate(df_eval["Questions"]):
        for interaction in log_dataset:
            if question.lower().translate(str.maketrans('', '', string.punctuation)) in interaction["question"].lower().translate(str.maketrans('', '', string.punctuation)):
                interaction["eval_question"]=question
                interaction["Ground Truth Answer"]= df_eval.at[index, "Ground truth answer"]
                interaction["Ground Truth Contexts"]= df_eval.at[index, "Relevant Context"]
    return log_dataset 

def doc_id_to_text(milvus_data, retrieval_json_data):
    """ In reranked_results, just have a string with chunk 1:.. chunk 2:
    get the contents of these chunks from milvus collection data"""
    
    for retrieval in retrieval_json_data:
        results= retrieval["reranked_results"].split("\n")
        matches= []
        for res in results: 
            match = re.search(r"'Doc_id':\s*'(\d+)'", res)
            if match:
                matches.append(match.group(1))
        retrieval["reranked_results"]=""
        for match in matches: 
            retrieval["reranked_results"]= retrieval["reranked_results"]+ '\n\n\n##' + milvus_data.loc[milvus_data["id"]==match, "text"].iloc[0]
        retrieval["eval_question"]=""
        retrieval["Ground Truth Answer"]= ""
        retrieval["Ground Truth Contexts"]= ""
    return retrieval_json_data

def userlog_to_json(path, collection_name, eval_path="./data/Evaluation dataset.xlsx"):
    df= pd.read_excel(path)
    df.drop(labels="retrievals", axis=1, inplace=True)
    retrieval_json= json.loads(df.to_json(orient="table"))
    milvus_data= get_data(collection=collection_name)
    retrieval_json_data= retrieval_json["data"]
    retrieval_log_dataset= doc_id_to_text(milvus_data=milvus_data, retrieval_json_data=retrieval_json_data)
    retrieval_log_dataset_complete= add_ground_truth(log_dataset=copy.deepcopy(retrieval_log_dataset), eval_path=eval_path)
    json_name= "/".join(path.split("/")[0:-1]) + '/' + path.split("/")[-1].split(".")[0]
    with open(f'{json_name}.json', 'w', encoding="utf8") as json_file:
        json.dump(retrieval_log_dataset_complete, json_file)


if __name__ == "__main__":
    userlog_to_json(parent_dir +"/user_data/user_temp/user_temp_retrieve.xlsx", collection_name="support_small", eval_path=parent_dir + "/data/Evaluation dataset.xlsx")