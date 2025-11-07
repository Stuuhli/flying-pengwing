import pandas as pd
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from app.config import API_GETCOLLECTION, DOC_NAME_METADATA  # noqa: E402
import requests  # noqa: E402

def extract_nested_values(row):
    """ flattens the nested dictionary

    Args:
        row (dict): each chunk from the db

    Returns:
        new_row: new chunk with flattened metadata
    """
    new_row= row.copy()
    for key1,item in row.items():
        if isinstance(item, dict):
            for key in list(item.keys()):
                if key==DOC_NAME_METADATA:
                    new_row["DocumentName"]= item[key]
                elif key=="Section":
                    new_row["Section"]= item[key]
                elif key=="Date":
                    new_row["Date"]= item[key]
                elif key=="page_no":
                    new_row["page_no"]= item[key]
                elif key=="binary_hash":
                    new_row["binary_hash"]= item[key]
            new_row.pop(key1)
    return new_row

def get_data(collection):
    response= requests.get(API_GETCOLLECTION.format(collection= collection))
    data_list= []
    if response.status_code==200:
        for i in response.json():
            row= (extract_nested_values(i))
            data_list.append(row)              
    data= pd.DataFrame(data_list)
    data["id"]= data["id"].astype("string")
    return data

if __name__ == "__main__":
    collection="support"
    data= get_data(collection=collection)
    data.to_excel(script_dir+f"/milvus_db_{collection}.xlsx", index=0)