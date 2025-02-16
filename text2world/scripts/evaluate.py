import os, sys
import pdb
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(ROOT_DIR)
import argparse
import pickle as pkl
from utils.world_generation import WorldGeneration
from utils.evaluator import Evaluator
import json
import pdb

sys.path.append('utils/llm')
sys.path.append('utils')
sys.path.append('analysis_script')

from parse_filename import parse_filename

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(ROOT_DIR)

import torch
import re


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.") # for llm
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    # parser.add_argument("--max_num_steps", required=False, default=30, help="specify the maximum number of steps used to finish the problems")
    parser.add_argument("--prompt_file_eval", required=True, default=None, help="specify the memory size")

    parser.add_argument('--prompt_file_gen', type=str, default='prompt/desc2domain')
    parser.add_argument('--prompt_style', type=str, choices=['generate', 'fillin'], default='generate')

    parser.add_argument("--gpu_num", required=False, default=0, help="specify the number of gpu cards", type=int)
    parser.add_argument("--save_path_gen", required=True, default=None, help="specify the name of llm agent", type=str)
    parser.add_argument("--save_path_eval", required=True, default=None, help="specify the name of llm agent", type=str)
    parser.add_argument("--llm_rating", required=True, default=None, help="specify the name of llm agent", type=bool)

    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')
    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')

    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def _purge_comments(pddl_str): # Purge comments from the given string.
    while True:
        match = re.search(r";(.*)\n", pddl_str)
        if match is None:
            return pddl_str
        start, end = match.start(), match.end()
        pddl_str = pddl_str[:start]+pddl_str[end-1:]
 
def convert_tensor_to_list(d): # for saving json file
    if isinstance(d, dict):
        return {k: convert_tensor_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensor_to_list(item) for item in d]
    elif isinstance(d, torch.Tensor):
        return d.tolist()
    else:
        return d

def annotate(args, data, evaluator):
    result_record = dict()
    result_record_0 = dict()
    
    for k, item in data.items():
        print(f'\033[92m======================= Annotating {k} =======================\033[0m')
        try:
            gt_without_comment, pred_domain = item['gt_domain_without_comment'], item['pred_domain']
            pred_0 = item['close_loop_world_generation_data']['correction_process']['0']['domain']
            result = evaluator.eval(gt_without_comment, pred_domain)
            result_0 = evaluator.eval(gt_without_comment, pred_0)
        except Exception as e:
            print('---- error ----')
            print(str(e))
            print('----')
            result = dict()

        result['id'] = item['id']
        result_0['id'] = item['id']
        
        result_record[f'{k}'] = result
        result_record_0[f'{k}'] = result_0
        
    return result_record, result_record_0


if __name__ == "__main__":
    args = parse()

    with open(args.save_path_gen, 'r') as file:
        data_dict = json.load(file)

    parsed_filename_part = parse_filename(os.path.basename(args.save_path_gen))
    MODEL = parsed_filename_part['MODEL']
    CORRECTION_TIME = parsed_filename_part['CORRECTION_TIME']
    DESCRIPTION_TYPE = parsed_filename_part['DESCRIPTION_TYPE']
    PROMPT_TYPE = parsed_filename_part['PROMPT_TYPE']

    evaluator = Evaluator(args) # init evaluator

    result_record, result_record_0 = annotate(args, data_dict, evaluator)

    save_dir = args.save_path_eval
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   

    result_record = convert_tensor_to_list(result_record) # json can not save torch data
    json.dump(obj=result_record, fp=open(os.path.join(args.save_path_eval, f'{MODEL}_{CORRECTION_TIME}_({DESCRIPTION_TYPE}-{PROMPT_TYPE}).json'), 'w', encoding='utf'), indent=4) # save as json file

    result_record_0 = convert_tensor_to_list(result_record_0) # json can not save torch data
    json.dump(obj=result_record_0, fp=open(os.path.join(args.save_path_eval, f'{MODEL}_0_({DESCRIPTION_TYPE}-{PROMPT_TYPE}).json'), 'w', encoding='utf'), indent=4) # save as json file