#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/home/ti514716/Pretrained/llava2

MODEL_NAME="instructblip"
MODEL_VERSION="Salesforce/instructblip-vicuna-7b"
BATCH_SIZE=2
SEED=242

guidance_strength_lst=(0.0 0.7)
TYPE=repro

BENCHMARK=chair
if [ $BENCHMARK == "chair" ]; then
    QUESTION_FILE_ls=(chair_coco_detr_th0.95_ram_th0.68.json)
elif [ $BENCHMARK == "pope" ]; then
    QUESTION_FILE_ls=(pope_coco_detr_th0.95_ram_th0.68.json)
fi

OUTPUT_DIR=./output/${MODEL_NAME}/answers/answer_${TYPE}_${BENCHMARK}

#### Generate answers ####
for guidance_strength in "${guidance_strength_lst[@]}"; do
    for QUESTION_FILE in "${QUESTION_FILE_ls[@]}"; do
        echo "Running $MODEL_VERSION inference with guidance_strength = $guidance_strength, seed = $SEED, batch_size = $BATCH_SIZE"
        python ./marine/generate_${MODEL_NAME}.py \
            --question_file $QUESTION_FILE \
            --guidance_strength $guidance_strength \
            --answer_path $OUTPUT_DIR \
            --model_path $MODEL_VERSION \
            --seed $SEED \
            --batch_size $BATCH_SIZE \
            --image_folder /home/ti514716/Data/coco/val2014 \
            --temperature 0.6 \
            --top_p 0.9 \
            --max_new_tokens 64
    done
done


#### EVALUATE ####
if [ $BENCHMARK == "chair" ]; then

    #### CHAIR EVALUATION ####
    echo "Running $MODEL_VERSION CHAIR metrics with seed = $SEED, batch_size = $BATCH_SIZE"

    python ./eval/format.py \
        --answer_dir $OUTPUT_DIR

    python ./eval/eval_chair.py \
        --eval_dir $OUTPUT_DIR \
        --save_path $OUTPUT_DIR/eval \

elif [ $BENCHMARK == "pope" ]; then

    #### POPE EVALUATION ####
    for QUESTION_FILE in "${QUESTION_FILE_ls[@]}"; do

        echo "Running $MODEL_VERSION pope evaluation with seed = $SEED, batch_size = $BATCH_SIZE"

        python ./eval/format.py \
            --answer_dir $OUTPUT_DIR

        python ./eval/eval_pope.py \
            --eval_dir $OUTPUT_DIR \
            --save_dir $OUTPUT_DIR/eval \
            --label_file $QUESTION_FILE \

    done
fi
