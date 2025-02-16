import os, sys
sys.path.append('utils/llm')
sys.path.append('utils')
ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(ROOT_DIR)
import re

import Levenshtein
import random
from datetime import datetime
from utils.pddl import parse_predicates, parse_actions, _checker, _purge_comments

class Evaluator:

    def __init__(self, args):
        self.args = args
        self.sim_model = None
        self.llm = None
        self.file_name = str(datetime.now()) + '_' + str(random.randint(1, 114514))

    def _init_llm(self):
        if self.llm is None:
            from get_agent import get_llm
            self.llm = get_llm(self.args)
            print('llm is loaded')

    def _init_llm(self):
        if self.llm is None:
            from get_agent import get_llm
            self.llm = get_llm(self.args)
            print('llm is loaded')

    def eval(self, gt_domain_text, pred_domain_text): # text
        executability = 0
        levenshtein_ratio_cleaned = 0
        action_f1_dict = 0
        predicate_f1_val = 0

    # f1   
        try:
            action_f1_dict = self.action_f1(gt_domain=gt_domain_text, pred_domain=pred_domain_text)
            predicate_f1_val = self.predicate_f1(gt_domain=gt_domain_text, pred_domain=pred_domain_text)
        except:
            pass
    # executability
        executability, text = _checker(pred_domain_text)
        print('exectability:', executability)

        if executability: # success
            print('the pred domain have no syntax error')
        else:
            pass

    # text distance
        try:
            gt_domain_text_cleaned = _purge_comments(gt_domain_text)
            pred_domain_text_cleaned = _purge_comments(pred_domain_text)
            # Remove all whitespace, newlines, tabs and other meaningless characters for PDDL
            gt_cleaned = re.sub(r'\s+', '', gt_domain_text_cleaned)
            pred_cleaned = re.sub(r'\s+', '', pred_domain_text_cleaned)
            levenshtein_ratio_cleaned = self.cal_Levenshtein_ratio(gt_cleaned, pred_cleaned)
        except:
            pass

        result = dict()
        result['executability'] = executability
        result['levenshtein_ratio_cleaned'] = levenshtein_ratio_cleaned
        # result['action_f1'] = action_f1_dict
        result['predicate_f1'] = predicate_f1_val
        result.update({'action_f1_'+k: v for k, v in action_f1_dict.items()})
        # Multiply all numeric values by 100 and round to 1 decimal place
        for key in result:
            if isinstance(result[key], (int, float)):
                result[key] = round(result[key] * 100, 1)
            elif isinstance(result[key], dict):
                for subkey in result[key]:
                    if isinstance(result[key][subkey], (int, float)):
                        result[key][subkey] = round(result[key][subkey] * 100, 1)
        return result
    
    def cal_Levenshtein_ratio(self, text1, text2):
        ratio = Levenshtein.ratio(text1, text2)
        return ratio

    def extract_rating(self, output):
        pattern = r"Rating: \[\[(\d+)\]\]" or r"\*\*Rating\*\*: \[\[(\d+)\]\]" or r"Rating\n \[\[(\d+)\]\]" or r"Rating\n\[\[(\d+)\]\]"

        match = re.search(pattern, output)
        if match:
            rating = int(match.group(1))
            print(f"The rating value is: {rating}")
            return rating
        else:
            print("No rating value found in the output.")
            return None

    def compute_f1_score(self, prediction, reference):
        if len(prediction) == 0 and len(reference) == 0:
            return 1
        pred_tokens = prediction
        ref_tokens = reference
        
        true_positives = len(set(pred_tokens) & set(ref_tokens))
        false_positives = len(set(pred_tokens) - set(ref_tokens))
        false_negatives = len(set(ref_tokens) - set(pred_tokens))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


    def predicate_f1(self, gt_domain, pred_domain):
        try:
            gt_pred_dict = parse_predicates(gt_domain)
            pred_pred_dict = parse_predicates(pred_domain)
            gt_predicates = [f"{k}_{v}" for k,v in gt_pred_dict.items()]
            pred_predicates = [f"{k}_{v}" for k,v in pred_pred_dict.items()]
            return self.compute_f1_score(pred_predicates, gt_predicates)
        except Exception as e:  # not executable
            print('error', e)
            return 0
        
    def _preprocess(self, l):
        l = map(str, l)
        l = sorted(l)
        for i in range(len(l)):
            if '(' == l[i][0] and ' or ' in l[i]:
                inner = l[i][1:-1].strip()
                parts = [p.strip() for p in inner.split(' or ')]
                parts = sorted(parts)
                l[i] = '(' + ' or '.join(parts) + ')'
        return sorted(l)


    def action_f1(self, gt_domain, pred_domain):    # param F1, preconds F1, effect F1
        def _mean(l):
            return sum(l) / len(l)
        metrics = {'params': [], 'preconds': [], 'effect': []}
        try:
            gt_actions = parse_actions(gt_domain)
            gt_actions = {k:{'params': self._preprocess([f"{p.symbol}" for p in v.parameters]),
                             'preconds':self._preprocess([x.strip() for x in str(v.precondition)[1:-1].split('and')]),
                             'effect':self._preprocess([str(x) for x in v.effects])} for k, v in gt_actions.items()}
            pred_actions = parse_actions(pred_domain)
            pred_actions = {k:{'params': self._preprocess([f"{p.symbol}" for p in v.parameters]),
                             'preconds':self._preprocess([x.strip() for x in str(v.precondition)[1:-1].split('and')]),
                             'effect':self._preprocess([str(x) for x in v.effects])} for k, v in pred_actions.items()}

            for k, gt_action in gt_actions.items():
                pred_action = pred_actions[k]
                metrics['params'].append(self.compute_f1_score(pred_action['params'], gt_action['params']))
                metrics['preconds'].append(self.compute_f1_score(pred_action['preconds'], gt_action['preconds']))
                metrics['effect'].append(self.compute_f1_score(pred_action['effect'], gt_action['effect']))
        except Exception as e:
            metrics['params'] = [0]
            metrics['preconds'] = [0]
            metrics['effect'] = [0]
        metrics = {k: _mean(v) for k, v in metrics.items()}
        return metrics
