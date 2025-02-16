#!/bin/bash

PROJECT_NAME=${1}
llm_rating=0  # 0 indicates False

for FILE in _generated_pddl/${PROJECT_NAME}/*.json; do
    FILENAME=$(basename -- "$FILE" .json) 
    echo "Processing file: $FILENAME"
    
    python text2world/scripts/evaluate_per_correction.py \
            --prompt_style generate   \
            --prompt_file_gen text2world/prompt/none \
            --cfg-path utils/text2world.yaml \
            --model gpt-4o  \
            --prompt_file_eval text2world/prompt/gpt4_evaluation \
            --save_path_gen "_generated_pddl/$PROJECT_NAME/$FILENAME.json" \
            --save_path_eval "_generated_pddl/_eval_result/per_correction/$PROJECT_NAME/$FILENAME.json" \
            --llm_rating "$llm_rating" \
            --gpu_num 0
done
