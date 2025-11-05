#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/ti514716/Pretrained/llava2

MODEL_NAME="instructblip"
MODEL_VERSION="Salesforce/instructblip-vicuna-7b"
guidance_strength_lst=(0.0 0.7)
BATCH_SIZE=24
TYPE=repro
SEED=242

set -a
source ./.env
set +a

benchmark_lst=(chair pope)
for BENCHMARK in "${benchmark_lst[@]}"; do
    # Initialize QUESTION_FILE_ls based on BENCHMARK
    if [ $BENCHMARK == "chair" ]; then
        QUESTION_FILE_ls=(chair_coco_detr_th0.95_ram_th0.68.json)
    elif [ $BENCHMARK == "pope" ]; then
        QUESTION_FILE_ls=(pope_coco_detr_th0.95_ram_th0.68.json)
    fi

    OUTPUT_DIR=./output/${MODEL_NAME}/answers/answer_${TYPE}_${BENCHMARK}
    
    #### EVALUATE ####
    if [ $BENCHMARK == "chair" ]; then

        #### CHAIR EVALUATION ####
        echo "Evaluating $MODEL_VERSION using CHAIR metrics with seed = $SEED, batch_size = $BATCH_SIZE, with eval dir = $OUTPUT_DIR"

        python ./eval/format.py \
            --answer_dir $OUTPUT_DIR

        python ./eval/eval_chair.py \
            --eval_dir $OUTPUT_DIR \
            --save_path $OUTPUT_DIR/eval \

    elif [ $BENCHMARK == "pope" ]; then

        #### POPE EVALUATION ####
        for QUESTION_FILE in "${QUESTION_FILE_ls[@]}"; do

            echo "Evaluating $MODEL_VERSION using POPE evaluation with seed = $SEED, batch_size = $BATCH_SIZE, with eval dir = $OUTPUT_DIR"

            python ./eval/format.py \
                --answer_dir $OUTPUT_DIR

            python ./eval/eval_pope.py \
                --eval_dir $OUTPUT_DIR \
                --save_dir $OUTPUT_DIR/eval \
                --label_file $QUESTION_FILE \

        done
    fi
done