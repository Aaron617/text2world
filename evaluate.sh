#!/bin/bash

PROJECT_NAME=${1}

llm_rating=0

if [ "$PROJECT_NAME" = "_all_gen" ]; then
    echo "Cannot create project with name '_all_gen'. Exiting..."
    exit 1
fi

for FILE in _generated_pddl/${PROJECT_NAME}/*.json; do
    FILENAME=$(basename -- "$FILE" .json)
    echo "Processing file: $FILENAME..."
    
    python text2world/scripts/evaluate.py \
            --prompt_style generate   \
            --prompt_file_gen text2world/prompt/None \
            --cfg-path utils/text2world.yaml \
            --model gpt-4o  \
            --prompt_file_eval text2world/prompt/gpt4_evaluation \
            --save_path_gen "_generated_pddl/$PROJECT_NAME/$FILENAME.json" \
            --save_path_eval "_generated_pddl/_eval_result/$PROJECT_NAME/" \
            --llm_rating "$llm_rating" \
            --gpu_num 0
done

cd analysis_script
python summerize_result.py $PROJECT_NAME