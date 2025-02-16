#!/bin/bash

MODEL=${1}
CORRECTION_TIME=${2}

PROMPT_TYPE="desc2domain_zeroshot_cot"
# PROMPT_TYPE="desc2domain_fewshot"

DESCRIPTION_TYPE="corrected_description"
# DESCRIPTION_TYPE="domain_description"

python text2world/scripts/generate.py \
        --data_path pddl_benchmark/our_benchmark_sample_20_modified.json   \
        --prompt_style generate   \
        --prompt_file_gen text2world/prompt/$PROMPT_TYPE    \
        --max_correction $CORRECTION_TIME  \
        --cfg-path utils/text2world.yaml    \
        --model $MODEL \
        --prompt_file_eval  text2world/prompt/gpt4_evaluation    \
        --save_path_gen _generated_pddl/_all_gen/$MODEL\_$CORRECTION_TIME\_\($DESCRIPTION_TYPE\-$PROMPT_TYPE\).json    \
        --gpu_num 0  \
        --description_type $DESCRIPTION_TYPE
