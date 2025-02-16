import pdb
import sys
import os
import re
import wandb
import warnings
import yaml
import json
import time
import argparse
from dotenv import load_dotenv
from llm import load_llm
from utils.logging.agent_logger import AgentLogger
from utils.logging.logger import SummaryLogger


logger = AgentLogger(__name__)
warnings.filterwarnings("ignore")

TASKS=["alfworld", "jericho", "pddl", "webshop", "webarena", "tool-query", "tool-operation", "babyai", "scienceworld", 'pddl-structured', 'babyai-train', 'neural', 'blockworld']


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, nargs='+',help="specify the tasks")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    parser.add_argument("--wandb", action="store_true", help="specify whether the wandb board is needed")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--project_name", required=False, default='', help="specify the project name for wandb")
    parser.add_argument("--baseline_dir", required=False, default='', help="specify the baseline loggings for wandb baseline comparison visualization")
    parser.add_argument("--max_num_steps", required=False, default=30, help="specify the maximum number of steps used to finish the problems")
    parser.add_argument("--gpu_num", required=False, default=0, help="specify the number of gpu cards", type=int)
    parser.add_argument("--engine", required=False, default=0, help="the path of model", type=str)
    # llm agent related
    parser.add_argument("--agent", required=False, default=None, help="specify the name of llm agent", type=str)
    parser.add_argument("--memory_size", required=False, default=None, help="specify the memory size", type=int)

    args = parser.parse_args()

    return args

def path_constructor(loader, node):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]

def load_config(cfg_path, args):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    llm_config = config["llm"]
    agent_config = config["agent"]
    env_config = config["env"]
    run_config = config["run"]
    
    if args.gpu_num > 0:
        llm_config['ngpu'] = args.gpu_num
    
    return llm_config, agent_config, env_config, run_config
  
def check_log_paths_are_ready(log_dir, baseline_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    if not os.path.exists(os.path.join(log_dir, "logs")):
        os.makedirs(os.path.join(log_dir, "logs"))
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    
    if not os.path.exists(os.path.join(log_dir, 'all_results.txt')):
        with open(os.path.join(log_dir, 'all_results.txt'), "w") as f:
            f.write("")
            f.close()

    return True

def get_llm(args):
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...
    # args = parse_args()
    llm_config, agent_config, env_config, run_config = load_config(args.cfg_path, args) 
    model = args.model
    # the model is trained based on another model
    if 'temp' not in model:
        llm_config = llm_config[args.model]
    else:
        llm_config = llm_config[model.split(':')[-1]] # base model
        print(llm_config)
        llm_config['engine'] = args.engine  # modify the path 
    if args.gpu_num > 0:
        llm_config['ngpu'] = args.gpu_num


    #---------------------------------------------- load llm -----------------------------------------------------
    logger.info("Start loading language model")
    llm = load_llm(llm_config["name"], llm_config)
    
    logger.info("Finished loading language model")
    
    return llm

if __name__ == "__main__":
    get_llm()