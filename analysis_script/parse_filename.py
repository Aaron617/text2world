import os, pdb
import yaml

def parse_filename(filename):
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)

    with open(parent_directory + '/../utils/text2world.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    base_name = os.path.splitext(filename)[0]
    model_list = data['llm'].keys()

    model = None
    for candidate in model_list:
        if base_name.startswith(candidate + "_"):
            model = candidate
            break
    
    if not model:
        raise ValueError(f"Model name in filename not found in the provided model list: {filename}")
    
    remaining = base_name[len(model) + 1:]
    
    try:
        correction_time, desc_prompt_part = remaining.split('_(', 1)
    except ValueError:
        raise ValueError("Filename format error: missing '_(' separator.")
    
    desc_prompt_part = desc_prompt_part.rstrip(')')
    try:
        description_type, prompt_type = desc_prompt_part.split('-', 1)
    except ValueError:
        raise ValueError("Missing hyphen separator in the part within parentheses.")
    
    return {
        'MODEL': model,
        'CORRECTION_TIME': correction_time,
        'DESCRIPTION_TYPE': description_type,
        'PROMPT_TYPE': prompt_type
    }