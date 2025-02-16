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

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(ROOT_DIR)

import torch
import copy
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

    parser.add_argument('--data_path', type=str)

    parser.add_argument("--gpu_num", required=False, default=0, help="specify the number of gpu cards", type=int)
    parser.add_argument("--save_path_gen", required=True, default=None, help="specify the name of llm agent", type=str)

    parser.add_argument('--max_correction', type=int, default=0)
    parser.add_argument('--description_type', type=str, default='corrected_description')

    # parser.add_argument('--stop_tokens', type=str, default='\n\n',
    #                     help='Split stop tokens by ||')
    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()

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

def annotate(args, data, world_generator):
    result_record = dict()
    
    for k, item in data.items():
        print(f'\033[92m======================= Annotating {k} =======================\033[0m')
        domain, description = item['pddl_domain'], item[args.description_type] # corrected_description, domain_description
        item['selected_description'] = description
        print(f'description {args.description_type}\n', f"\033[94m{description}\033[0m", '\n')
        record = dict()
        gt_without_comment, pred, success, generated_data = None, None, None, None
        while pred is None:
            try:
                success, generated_data = world_generator.close_loop_world_generation(item)
                gt, pred = domain, generated_data['pred_domain']
                gt_without_comment = _purge_comments(gt)
                record['error'] = None
            except Exception as e:
                print(str(e))
                record['error'] = str(e)

        record['id'] = item['id']
        record['gt_domain_raw'] = domain
        record['gt_domain_without_comment'] = gt_without_comment
        record['pred_domain'] = pred
        record['domain_description'] = description
        record['close_loop_world_generation_success'] = success
        record['close_loop_world_generation_data'] = generated_data

        result_record[f'task_{k}'] = record

        tmp_result_record = copy.deepcopy(result_record)
        save(tmp_result_record, args)

        tmp_result = convert_tensor_to_list(result_record) # json can not save torch data
        save(tmp_result, args)

    return result_record

def save(result_record, args):
    result_record = convert_tensor_to_list(result_record) # json can not save torch data
    save_dir = os.path.dirname(args.save_path_gen)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    json.dump(obj=result_record, fp=open(args.save_path_gen, 'w', encoding='utf'), indent=4) # save as json file

if __name__ == "__main__":
    args = parse()
    
    world_generator = WorldGeneration(args) # init world generator

    # load domain - description data
    data = pkl.load(open(args.data_path, 'rb')) if 'pkl' in args.data_path else json.load(open(args.data_path))
    if type(data) == list:
        data = {idx:_ for idx, _ in enumerate(data)}

    result_record = annotate(args, data, world_generator)
    save(result_record, args)

    
    
