import json, pdb
from tabulate import tabulate
import os
from parse_filename import parse_filename
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name")
    args = parser.parse_args()
    return args


def _mean(l):
    if l == []:
        return 0
    return sum(l) / len(l)

args = parse()

directory = f"../_generated_pddl/_eval_result/{args.project_name}"

table_data = []

headers = [
    "Model",  "Correction Time", "Description Type", "Prompt Type", "Executability", "Domain Text Similarity",
    "Predicate F1", "Action F1 Params",
    "Action F1 Preconds", "Action F1 Effect"
]
eval_files = os.listdir(directory)
eval_files.sort()

if os.path.exists(f"../_generated_pddl/_eval_result/{args.project_name}/_result_board.txt"):
    os.remove(f"../_generated_pddl/_eval_result/{args.project_name}/_result_board.txt")

for filename in eval_files:
    parsed_filename_part = parse_filename(filename)
    MODEL = parsed_filename_part['MODEL']
    CORRECTION_TIME = parsed_filename_part['CORRECTION_TIME']
    DESCRIPTION_TYPE = parsed_filename_part['DESCRIPTION_TYPE']
    PROMPT_TYPE = parsed_filename_part['PROMPT_TYPE']
    try:
        executability, domain_text_similarity, llm_rating, distance, distance_cleaned, action_f1, predicate_f1 = 0, 0, 0, 0, 0, dict(), 0
        action_f1['params'], action_f1['preconds'], action_f1['effect'] = 0, 0, 0
        with open(os.path.join(directory, filename), 'r') as f:
            data = json.load(f)
            cnt = 0
            for key in data.keys():
                result = data[key]
                assert result is not None, "result is none"
                # for k, v in result['action_f1'].items():
                #     action_f1[k] += _mean(v)
                action_f1['params'] += result['action_f1_params']
                action_f1['preconds'] += result['action_f1_preconds']
                action_f1['effect'] += result['action_f1_effect']
                executability += result['executability']
                # try:
                #     domain_text_similarity += result['domain_text_similarity'][0][0] 
                # except:
                #     domain_text_similarity += 0
                try:
                    domain_text_similarity += result['levenshtein_ratio_cleaned']
                except:
                    domain_text_similarity += 0
                # if result['llm_rating'] != 0 and result['llm_rating'][0] != None:
                #     try:
                #         llm_rating += result['llm_rating'][0]
                #     except:
                #         llm_rating += 0
                try:
                    distance += result['distance']
                except:
                    distance += 0
                try:
                    distance_cleaned += result['distance_cleaned']
                except:
                    distance_cleaned += 0
                predicate_f1 += result['predicate_f1']
                cnt += 1

            episode = cnt
            
            executability = round(executability/episode, 1)
            domain_text_similarity = round(domain_text_similarity/episode, 1)
            # llm_rating = round(llm_rating/episode, 2)
            # distance = round(distance/episode, 2)
            # distance_cleaned = round(distance_cleaned/episode, 2)
            predicate_f1 = round(predicate_f1/episode, 1)
            for k in action_f1.keys():
                action_f1[k] = round(action_f1[k]/episode, 1)
            
            table_data.append([
                MODEL,
                CORRECTION_TIME, 
                DESCRIPTION_TYPE,
                PROMPT_TYPE,
                executability,
                domain_text_similarity,
                # llm_rating,
                # distance,
                # distance_cleaned,
                predicate_f1,
                action_f1['params'],
                action_f1['preconds'],
                action_f1['effect']
            ])
    except:
        print("error file: ", filename)

table_data.sort(key=lambda x: x[0])

table = tabulate(table_data, headers, tablefmt="pretty")

with open(f"../_generated_pddl/_eval_result/{args.project_name}/_result_board.txt", "w") as file:
    file.write(table)