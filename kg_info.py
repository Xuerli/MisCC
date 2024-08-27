import argparse
from tqdm import tqdm
from pathlib import Path
import argparse
import rdflib
import re
# import urllib3.request
import os
import requests
import urllib.parse
import json
import time
import sys
import pandas as pd
import pickle as pkl

from tqdm import tqdm
import ollama
import queryLlama3 as ql
import pdb
import ollama
import warnings
import queryLlama3 as ql
from datetime import datetime
import pdb
warnings.filterwarnings('ignore') # setting ignore as a parameter

# send SPARQL query to wikiData and return the answer
def ansWiki(query):
    try:
        # set an agent to avoid 403 HTTP ERROR
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                               agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        sparql.setTimeout(60)  # 5 minutes
        # Get the answering {'head': {'vars': [...]}, 'results': {'bindings': [...]}}
        ans = sparql.query().convert()
    except:
        ans = {'results': {'bindings': []}}

    return ans


# find all paths between a member of fromIds to a member of toIds, where the paths length is smaller than the DepthLimit
# return a list of dictionaries representing shortest paths
def get_path(fromIds, toIds):
    i = 0
    DepthLimit = 3
    output = []
    # pdb.set_trace()
    duplicates = list(set(fromIds).intersection(toIds))
    # one-node path for one entity to itself with the minimal path length as 0
    for x in duplicates:
        mapInfo = {'from': x, 'to': x, 'paths': [[x]]}
        output.append(mapInfo)

    # otherwise, find closest path with sparql query
    restToIds = [id for id in toIds if id not in duplicates]
    while i < DepthLimit and restToIds != []:
        queryHead = "SELECT ?from ?to"
        fromValues = ' wd:'.join(fromIds)
        toValues = ' wd:'.join(restToIds)
        queryValues = f"\n VALUES ?from {{ wd:{fromValues} }} \n VALUES ?to {{ wd:{toValues} }} "
        queryMain = '\n' + 'WHERE\n{' + queryValues + '\n ?from'
        for j in range(i):
            # pair the varialbes of the relation and the objection
            pair = '?r' + str(j) + ' ' + '?id' + str(j)
            queryHead += pair
            # the current objection will be the subjection of the next triple, placed after '\n'
            queryMain += ' ' + pair + '.\n  ' + ' ?id' + str(j)

        queryHead += ' ?r' + str(i)
        queryMain += ' ?r' + str(i) + '?to.'
        queryEnd = '\n}'
        query = queryHead + queryMain + queryEnd
        result = ansWiki(query.encode('utf-8'))['results']

        checkedToIds = []
        # Check the answer
        if result['bindings'] == []:
            i += 1  # continue to the next step
        else:
            # store found paths in a list.
            for rawList in result['bindings']:  # rawList = {{id0...}, {id1...},...{idj...}, {r0...},...{rj+1}}
                paths = []  # initialise the output lists as empty
                startId = rawList['from']['value'].split('/')[-1]
                toId = rawList['to']['value'].split('/')[-1]
                checkedToIds.append(toId)
                path = startId + ' '
                # Construct the path by appending the jth triple. Each triple are separated by a comma.
                for j in range(i):
                    # pair the varialbes of the relation and the objection
                    relation = rawList['r' + str(j)]['value'].split('/')[-1]
                    objId = rawList['id' + str(j)]['value'].split('/')[-1]
                    # the object of this triple will be the subject of the next triple
                    path += (relation + ' ' + objId + ', ' + objId + ' ')
                path += (rawList['r' + str(i)]['value'].split('/')[-1] + ' ' + str(toId))
                paths.append(path.split(', '))
                mapInfro = {'from': startId, 'to': toId, 'paths': paths}
                if mapInfro not in output:
                    output.append(mapInfro)
            i += 1
            restToIds = [id for id in restToIds if id not in checkedToIds]

    return output


# calculate the semantic paths and their length
def semanticPath(pathsDicList, logRList, diffRList, sims):
    pathNewList = []

    for pathDic in pathsDicList:
        newPathDic = pathDic.copy()
        pathList = pathDic['paths']
        newPathList = []

        singleNodePath = 0
        for path in pathList:
            newPath = []
            invalid = 0
            for triple in path:
                # ignore logical relation by remove the step
                if ' ' not in triple:
                    singleNodePath = 1
                    newPath.append(triple)
                elif triple.split()[1] in diffRList:
                    invalid = 1
                    #print('\n   invalid  \n' + str(triple))
                elif triple.split()[1] not in logRList:
                    newPath.append(triple)
            # no different from
            if invalid != 1:
                # have found the shortest path: [].
                if newPath == []:
                    newPathList = [[]]
                    break
                elif newPathList == [] or len(newPathList[0]) > len(newPath):
                    newPathList = [newPath]

        newPathDic['newPaths'] = sorted(newPathList)
        if newPathList != []:
            if singleNodePath == 1:
                newPathDic['minpathLen'] = 0
            # adjustment
            elif [] in  newPathDic['newPaths']:
                newPathDic['minpathLen'] = sims
            else:
                newPathDic['minpathLen'] = len(newPathDic['newPaths'][0])
        else:
            newPathDic['minpathLen'] = 4
        if newPathDic not in pathNewList:
            pathNewList.append(newPathDic)

    return pathNewList

# column_in is either 'PIds' or 'CIds'
def kg_info_extract(row, input_columns, filesRecord):
    [log_r_list, diff_r_list] = filesRecord
    Antecedent_kg = ''
    Consequent_kg = ''
    valid_entities = []
    PIds = input_columns[0]
    CIds = input_columns[1]
    invalid = ['nan', 'na', '', '[]']
    invalid_num = 0
    llama3_Conf_PIds = eval(row[PIds])

    if str(llama3_Conf_PIds).lower() not in invalid:
        valid_entities = [x['ID'] for x in llama3_Conf_PIds]
        for entity in llama3_Conf_PIds:
            Antecedent_kg += 'The usual meaning of {} is that {} '.format(entity['label'], entity['description'])
    # pdb.set_trace()

    llama3_Conf_CIds = eval(row[CIds])
    if str(llama3_Conf_CIds).lower() not in invalid:
        valid_entities += [x['ID'] for x in llama3_Conf_CIds]
        for entity in llama3_Conf_CIds:
            Consequent_kg += 'The usual meaning of {} is that {}.'.format(entity['label'], entity['description'])

    fromIds = [x['ID'] for x in llama3_Conf_PIds]
    toIds = [x['ID'] for x in llama3_Conf_CIds]
    # Get kg paht information
    paths_list = get_path(fromIds, toIds)
    semPaths= semanticPath(paths_list, log_r_list, diff_r_list, 0)
    valid_SemPaths_cand = [x['newPaths'] for x in semPaths if x['from'] in valid_entities or x['to'] in valid_entities]

    valid_SemPaths = []
    for pathsLists in valid_SemPaths_cand:
        # pathsLists = [['Q2 P527 Q3230', 'Q3230 P186 Q627']]
        for pathList in pathsLists:
            for path in pathList:
                if ' ' in path:
                    valid_SemPaths.append(path)
    return Antecedent_kg, Consequent_kg, semPaths, valid_SemPaths

def run(dataframe, out_file, out_columns, input_columns, filesRecord):
    out_file_csv = out_file + '.csv'
    number_lines = len(dataframe)
    chunksize = 12

    if (out_file is None):
        out_file_valid = False
        already_done = pd.DataFrame().reindex(columns=dataframe.columns)
        start_line = 0

    elif isinstance(out_file_csv, str):
        out_file_valid = True
        if os.path.isfile(out_file_csv):
            already_done = pd.read_csv(out_file_csv)
            start_line = len(already_done)
        else:
            already_done = pd.DataFrame().reindex(columns=dataframe.columns)
            already_done[out_columns] = ''
            start_line = 0
    else:
        print('ERROR: "out_file_csv" is of the wrong type, expected str')

    for i in tqdm(range(start_line, number_lines, chunksize)):
        sub_df = dataframe.iloc[i: i + chunksize]
        # print('\n\n\nsub_df is \n'+str(sub_df))
        sub_df[out_columns] = sub_df.apply(lambda x: kg_info_extract(x, input_columns, filesRecord), axis=1, result_type='expand')
        already_done = pd.concat([already_done, sub_df], axis=0)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_csv(out_file)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_pickle(out_file.replace(".csv", '.pkl')
    return already_done

if __name__ == "__main__":
    # Retrieve arguments
    # argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument("input_file", type=Path, action="store",
    #                              help="The input file with raw lists of entity IDs")
    # argument_parser.add_argument("out_file", type=Path, action="store",
    #                              help="The output file with the retrieved KG property list")
    #
    # args = argument_parser.parse_args()
    # input_file = args.input_file
    # out_file = args.out_file

    input_file = 'data/miscc_tv_both_kg_llama3.pkl'
    # input_file = 'test.csv'
    out_file = 'data/miscc_tv_both_kg_info.pkl'
    
    if input_file[-4:] == '.pkl':
        df = pd.read_pickle(input_file)
    elif input_file[-4:] == '.csv':
        df = pd.read_csv(input_file)
    else:
        print("invalid input file")

    print(df.columns)
    entity_info_file = ("files_record/llama3_entity_selection_log.json")
    kg_prop = ("files_record/propertyCatagories.txt")
    fproperty = open(kg_prop)
    [logRelation, diffRelation] = fproperty.read().splitlines()
    log_r_list = logRelation.split(' ')
    diff_r_list = diffRelation.split(' ')
    filesRecord = [log_r_list, diff_r_list]
    input_columns = ['llama3_Conf_PIds', 'llama3_Conf_CIds']
    out_columns = ['Antecedent_kg', 'Consequent_kg', 'semPaths', 'valid_SemPaths']

    run(df.loc[:, ~df.columns.str.contains('^Unnamed')], out_file, out_columns, input_columns, filesRecord)
    print('finished')
