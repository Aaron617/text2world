import json
import os
# from utils.openai_access import Generator
from utils.pddl import extract_pddl, extract_domain_name, _checker
from tqdm import tqdm
import random
import time, pdb
from datetime import datetime

import os, sys
sys.path.append('utils/llm')
sys.path.append('utils')
from get_agent import get_llm

correction_format = '''
Round [Round]
Incorrect PDDL:
[PDDL]
Error Information:
[Error]
Corrected PDDL:
[Corrected_PDDL]
'''

history_format = '''
Round [Round]
Incorrect PDDL: 
[PDDL]
Error Information:
[Error]
'''

def make_correction_prompt(prompt, trace, domain, error_info):
    history = [history_format.replace('[Round]', str(idx)).replace('[PDDL]', _['incorrect_domain']).replace('[Error]', _['error_info']) for idx, _ in enumerate(trace)]
    history = history[-5:]
    history.append(correction_format.replace('[Round]', str(len(trace))).replace('[PDDL]', domain).replace('[Corrected_PDDL]', '').replace('[Error]', error_info))
    prompt += '\n\n' + '\n\n'.join(history)
    return prompt


domain_correct_prompt = '''
I would like you to serve as an expert in PDDL, assisting me in correcting erroneous PDDL code. I will provide you with the incorrect PDDL along with the error messages returned by the system. You should output your thought process firstly. You MUST enclose the COMPLETE corrected PDDL within ```pddl```.
Here are some hints to help you debug the pddl domain file:
1. You should start by checking if all the essential domain constructs or domain definition constructs are present. Commonly included domains comprise:
    a. :domain declaration to name the domain.
    b. :requirements to specify the PDDL features used in the domain.
    c. :types to define any object types for categorizing entities in the planning problem.
    d. :constants (if necessary) to declare any objects that remain static throughout the planning problems.
    e. :predicates to define the properties and relations between objects that can change over time.
    f. :functions (if necessary) to define numeric functions for more complex domains.
    g. :action definitions for each action that agents can perform, including parameters, preconditions, and effects.
2. You need to check the number of parameters of each actions.
3. Having :typing in the domain indicates that it uses a hierarchy to organize objects. Therefore, it's crucial to clearly list all object types related to the planning task in a :types section.
4. '-' should not appear in :types.
'''


class WorldGeneration:
    def __init__(self, args) -> None:
        self.generator = get_llm(args)
        self.prompt_file = args.prompt_file_gen
        self.prompt_style = args.prompt_style
        self.max_correction = args.max_correction
        self.model = args.model

    def _make_traj_prompt(self, traj):
        traj_prompt = []
        for example in traj['trajectory']:
            if 'action_text' in example:
                act = example['action_text']
                traj_prompt.append(f'Action: {act}')
            obs = example['state_text']
            traj_prompt.append(f'State: {obs}')
        return '\n'.join(traj_prompt)

    def _make_domain_generation_prompt(self, data):
        prompt_template = open(self.prompt_file).read()
        description = data['selected_description']
        if self.prompt_style == 'generate':
            prompt = prompt_template.replace('[Description]', description)
        elif self.prompt_style == 'fillin':
            assert 'fillin' in self.prompt_file
            prompt = prompt_template.replace('[Description]', description).replace('[Draft]', data['unfilled_domain']).replace('[Traj]', self._make_traj_prompt(data['example_trajectory']))
        else:
            raise Exception(f'{self.prompt_style} Not Implemented')
        return prompt

    def _domain_generation(self, prompt):
        success = False
        while not success:
            success, gpt_response = self.generator.generate(system_message="you are a helpful assistant", prompt=prompt)
            print(gpt_response)
        tokens = 0
        
        try:
            domain_pddl = extract_pddl(gpt_response)
            error_info = None
        except Exception as e:
            print(f'Error extracting PDDL: {e}')
            domain_pddl = gpt_response
            error_info = str(e)
        return domain_pddl, error_info

    def _domain_correction(self, domain, init_error_info=None, max_retry=3):
        domain_name = extract_domain_name(domain)
        count = 0
        trace = []
        token = 0
        correction_process = dict()
        prev_error_info = init_error_info
        while True:
            if prev_error_info is None:
                success, error_info = _checker(domain)
            else:
                success, error_info = False, prev_error_info
            correction_process[f'{count}'] = {'domain_name': domain_name, 'domain': domain}
            if success:
                print(f'Env {domain_name} Passed Test')
                return success, domain, trace, token, correction_process
            elif count == max_retry:
                print(f'Env {domain_name} Retry Exceeded')
                return False, domain, trace, token, correction_process
            else:
                print(f'Env {domain_name}, Correct Round {count}. Error Info: {error_info}')
                count += 1
                prompt = domain_correct_prompt
                prompt = make_correction_prompt(prompt, trace, domain, error_info)
                try:
                    success, text = self.generator.generate(system_message="you are a helpful assistant", prompt=prompt)
                    tokens = 0 # TODO
                    pre_domain = domain
                    try:
                        domain = extract_pddl(text)
                        prev_error_info = None
                    except Exception as e:
                        domain = text
                        prev_error_info = str(e)
                    trace.append({'error_info': error_info, 'incorrect_domain': pre_domain, 'corrected_domain': domain, 'gpt_response': text, 'prompt': prompt})
                except Exception as e:
                    trace.append({'error_info': error_info, 'incorrect_domain': pre_domain, 'corrected_domain': domain, 'gpt_response': text, 'prompt': prompt})
        
        return False, domain, trace, token, correction_process


    def close_loop_world_generation(self, data):
        '''
            data: data is a dictionary contains "description" (mandatory), "unfilled_domain", "example_trajectory" (optional, only under "fillin" setting), 
        '''
        st = time.time()
        prompt = self._make_domain_generation_prompt(data)
        domain, error_info = self._domain_generation(prompt)
        success, corrected_env, trace, token, correction_process = self._domain_correction(domain, init_error_info=error_info, max_retry=self.max_correction)
        return success, {'pred_domain': corrected_env, 'correct_trace':trace, 'time': time.time() - st, 'first_prompt': prompt, 'correction_process': correction_process}
