from datetime import date
import json
from json.decoder import JSONDecodeError
from datetime import datetime


def write_jsonLog(data, file):

    with open(file, mode='a+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    return


def read_jsonLog(prompting, file):

    with open(file, mode='r', encoding='utf-8') as f:
        record_str = f.read()
        f.close()
        record_list = json.loads("[" + record_str.replace("}{", "},{") + "]")
        for record in record_list:
            if record['query'] == prompting:
                return record
    return {}


def query(query_p, model, tokenizer, device):
    today = datetime.today().strftime('%Y-%m-%d')
    print(query_p)
    try:
        model_inputs = tokenizer(query_p, return_tensors = "pt", padding=True).to("cuda")
        model_inputs.to(device) 
        input_length = model_inputs.input_ids.shape[1]
        
        # # Generate
        generated_ids = model.generate(**model_inputs, max_new_tokens=256)
        full_answer1 = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(full_answer1)
        query_record = {"query": query_p, 
                        "response": full_answer1,
                        "date": today}
        try:
            output = json.loads(full_answer1)
            query_record["Answer"] =  output["Answer"]
            query_record["Explanation"] =  output["Explanation"]
        except JSONDecodeError:
            query_record["Answer"] =  ""
            query_record["Explanation"] =  ""
    except:
        # print('Exception in query Llama3')
        query_record = {"query": query_p, 
                        "response": "",
                        "date": today,
                        "Answer":"",
                        "Explanation":""}
        
    return output
