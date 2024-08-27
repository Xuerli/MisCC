import argparse
from tqdm import tqdm
import rdflib
import re
# import urllib3.request
import os
import requests
import urllib.parse
import json
from datetime import datetime
import sys
import pandas as pd
import pickle as pkl

from tqdm import tqdm
import ollama
import warnings
import queryLlama3 as ql
import pdb

# column_in is either 'PIds' or 'CIds'
def kg_info_extract(row, relation, record_list, record_file_name):
    unknown = True
    today = datetime.today().strftime('%Y-%m-%d')
    new_entity = {}
    invalid_num = 999
    llama3_answer = invalid_num
    prompting_template = '''Is the wikidata entity Entity mentioned in the claim Claim1? Context1 is the online conversation about climate change where Claim1 came from. Answer in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure.\n Entity: {} Entity's \n Claim1: {}\n Context1: {}'''
    if 'CC' in row.keys():
        context = row['CC']
    else:
        context = "It is potentially from a conversation of hate speech"

    output = []

    for key in relation.keys():
        # pdb.set_trace()
        entities = []
        for entityId in eval(row[relation[key]]):
            # pdb.set_trace()
            if None == re.search("^Q", entityId):
                print('not an entity :' + entityId)
                continue
            # print('entity ', entityId)
            #  if it is an existing query, just retrieve the answer from the record
            for record in record_list:
                if 'entity' in record.keys() and set(record['entity'].keys()) == {'ID', 'label', 'description'} and record['entity']['ID'] == entityId:
                    entityInfo = record['entity']

                    query_p = prompting_template.format(entityInfo, row[key], context)
                    if query_p == record['query']:
                        llama3_answer = record["respond"]
                        unknown = False
                        break
            #  otherwise, query llama3 and extend the record with this new query.
            if unknown:
                with urllib.request.urlopen(
                        "https://www.wikidata.org/w/api.php?action=wbgetentities&ids=" + entityId + "&format=json") as url:
                    data = json.loads(url.read().decode())
                    try:
                        description = data['entities'][entityId]['descriptions']['en']['value'],
                        label = data['entities'][entityId]['labels']['en']['value']
                        entityInfo = {'ID': entityId,
                                      'description': description,
                                      'label': label}
                        query_p = prompting_template.format(entityInfo, row[key], context)
                        llama3_answer = ql.query(query_p, record_file_name)
                        if "Answer" in llama3_answer.keys() and llama3_answer["Answer"] != invalid_num:
                            query_record = {"entity": entityInfo, "claim": row[key], "query": query_p, "respond": llama3_answer, "date": today}
                            record_list.append(query_record)
                    except:
                        # print('Error: no llama3 answer.')
                        llama3_answer = {"Answer": 0, "Explanation": "Failed in getting llama3's answer so being set as the default value of unknown"}

            # print('Llama3 responds: ', llama3_answer)
            if "Answer" in llama3_answer.keys() and int(llama3_answer["Answer"]) == 1:
                entities.append(entityInfo)
                # print('appended entity', entityInfo)
        output.append(entities)
    print(output, len(output))
    return output



def run(dataframe, out_file, out_columns, function, relation, record_list, record_file_name):
    number_lines = len(dataframe)
    chunksize = 1

    if (out_file is None):
        out_file_valid = False
        already_done = pd.DataFrame().reindex(columns=dataframe.columns)
        start_line = 0

    elif isinstance(out_file, str):
        out_file_valid = True
        if os.path.isfile(out_file):
            already_done = pd.read_csv(out_file)
            start_line = len(already_done)
        else:
            already_done = pd.DataFrame().reindex(columns=dataframe.columns)
            start_line = 0
    else:
        print('ERROR: "out_file" is of the wrong type, expected str')

    for i in tqdm(range(start_line, number_lines, chunksize)):
        sub_df = dataframe.iloc[i: i + chunksize]
        sub_df[out_columns] = sub_df.apply(lambda x: function(x, relation, record_list, record_file_name), axis=1, result_type='expand')
        already_done = pd.concat([already_done, sub_df], axis=0)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_csv(out_file)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_pickle(out_file.replace(".csv", '.pkl'))
    return already_done

# Set the API key for the Chat GPT

# get rid of '\r'
def get_rid(input):
    df = pd.read_csv(input, encoding='UTF-8-SIG', on_bad_lines='skip', lineterminator='\n')
    df.columns = [*df.columns[:-1], str(df.keys()[-1]).replace('\r', '')]
    for i in range(0, len(df)):
        df.at[i, df.keys()[-1]] = str(df.iloc[i][df.keys()[-1]]).replace('\r', '')
    df.to_csv(input)

if __name__ == "__main__":
    input_file = 'data/multitarget_KN_grounded_CN_kg.pkl'
    # input_file = 'test.csv'
    out_path = 'data/multitarget_KN_grounded_CN_kg_entities.csv'
    # get_rid(input_file)
    if input_file[-4:] == '.pkl':
        df = pd.read_pickle(input_file)
    elif input_file[-4:] == '.csv':
        df = pd.read_csv(input_file)
    else:
        print("invalid input file")

    # df.to_csv('data/miscc_tv_both_kg.csv')
    print("there are {} ccs with columns {}: ".format(len(df), df.columns))
    record_file_name = ("/Users/xueli/Library/CloudStorage/OneDrive-UniversityofEdinburgh/code/miscc_1/kg_entity_linking/files_record/llama3_entity_selection_log_Wendy.json")
    data_log = open(record_file_name, mode='a+', encoding='utf-8')
    record_str = data_log.read()
    data_log.close()
    record_list = json.loads("[" + record_str.replace("}{", "},{") + "]")
    relation = {'knowledge_sentence': 'PIds', 'hate_speech': 'CIds'}
    out_columns = ["_".join(['llama3_Conf', x, relation[x]]) for x in relation.keys()]
    print('out_columns: {}'.format(out_columns))
    run(df.loc[:, ~df.columns.str.contains('^Unnamed')], out_path, out_columns, kg_info_extract, relation,  record_list, record_file_name)
    print('finished')
    # run(df.loc[:, ~df.columns.str.contains('^Unnamed')], out_path, 'llama3_singleP', queryllama)

    print('finished')