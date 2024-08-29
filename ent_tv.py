# Cross Validation Classification LogLoss
import pandas as pd
from tqdm import tqdm
import os
import queryLlama3 as ql
import pdb
import warnings
import itertools
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import huggingface_hub
from transformers import pipeline
warnings.filterwarnings('ignore') # setting ignore as a parameter

def model(name, device):
    huggingface_hub.login(token='') # add your user token here
    model = LlamaForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def combinations(row):
    premises = [str(row['Antecedent']),
                ' '.join([str(row['Antecedent']), str(row['Antecedent_kg'])]),
                ' '.join([str(row['Antecedent']), str(row['Consequent_kg'])]),
                ' '.join([str(row['Antecedent']), str(row['Antecedent_kg']), str(row['Consequent_kg'])])]
    conclusions =[ row['Consequent'],
                    ' '.join([row['Consequent'], str(row['Antecedent_kg'])]),
                    ' '.join([row['Consequent'], str(row['Consequent_kg'])]),
                    ' '.join([row['Consequent'], str(row['Antecedent_kg']), str(row['Consequent_kg'])])]
    return  premises, conclusions

def queryCCC(row, arguments):
    [template, model, tokenizer, device] = arguments
    premises, conclusions = combinations(row)
    sub_claim_pairs = list(itertools.product(premises, conclusions ))
    queries = [template.format(x[0], x[1]) for x in sub_claim_pairs]
    print(len(queries), queries)

    output = []
    for q in queries:
        respond = ql.query(q, model, tokenizer, device)
        output += respond
    
    return output


def run(dataframe, out_file, out_columns, function, arguments):
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

    # for i in tqdm(range(start_line, number_lines, chunksize)):
    for i in tqdm(range(start_line, 2, 2)):
        sub_df = dataframe.iloc[i: i + chunksize]
        sub_df[out_columns] = sub_df.apply(lambda x: function(x, arguments), axis=1, result_type='expand')
        # pdb.set_trace()
        already_done = pd.concat([already_done, sub_df], axis=0)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_csv(out_file)
        already_done.loc[:, ~already_done.columns.str.contains('^Unnamed')].to_pickle(out_file.replace(".csv", '.pkl'))

    return already_done


if __name__ == '__main__':
    input_file = 'data/miscc_tv_both_kg_info.csv'
    # input_File = '/Users/xueli/Library/CloudStorage/OneDrive-UniversityofEdinburgh/code/miscc_1/2024_data/miscc_data/baseline/semval_classification_2.csv'
    out_path = 'data/miscc_tv_both_kg_info_tv.csv'

    if input_file[-4:] == '.pkl':
        df = pd.read_pickle(input_file)
    elif input_file[-4:] == '.csv':
        df = pd.read_csv(input_file)
    else:
        print("invalid input file")
    device = 0 if torch.cuda.is_available() else -1
    
    model, tokenizer = model("meta-llama/Meta-Llama-3-8B", device)
   
    prompt = '''Question: decide whether Claim2 follows Claim1, i.e. Claim1 entails or infers Claim2.
            Claim1: You are a boomer.
            Claim2: You understand.
            Answer the following question in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
                
    model_inputs = tokenizer(prompt, return_tensors = "pt", padding=True).to("cuda")
    model_inputs.to(device) 
    input_length = model_inputs.input_ids.shape[1]
    
    # # Generate
    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    x = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print('\n',x,'\n88888888888888888888\n\n')
    

#     template_vef = '''A claim is verifiable if its truth value can be derived or tested to be true or false based on specified knowledge. Is Claim1 verifiable? Claim1 is originally from the CC in an online conversion. Claim1: {} 
#     CC: {}
#     Answer in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure. '''
#     out_columns_vef_ant = ['Vf_Ant' +str(i) for i in range(13)]
#     out_columns_vef_con = ['Vf_Con' +str(i) for i in range(13)]
    
#     template_sc_tv ='''Claim1 is originally from the Claim2 from an online conversion. Is Claim1 true?  
# Claim1: {}
# Claim2: {}
# Answer in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure.'''
#     out_columns_tv_ant = ['TV_Ant_' +str(i) for i in range(13)]
#     out_columns_tv_con = ['TV_Con_' +str(i) for i in range(13)]
    
    template_ent = '''Question: decide whether Claim2 follows Claim1, i.e. Claim1 entails or infers Claim2.
            Claim1: {}.
            Claim2: {}.
            Answer the following question in JSON format with fields "Answer" and "Explanation". In "Answer", use 1 for Yes or -1 for No or 0 for unsure.'''
    out_columns_ent = []
    for i in range(16):
        out_columns_ent += ['Ent_query_' + str(i), "Ent_Answer" + str(i), "Ent_Explanation" + str(i)]

    run(df.loc[:, ~df.columns.str.contains('^Unnamed')], out_path, out_columns_ent, queryCCC, [template_ent, model, tokenizer, device])

