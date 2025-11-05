from datetime import datetime
import os
import sys
import traceback
sys.path.append(os.getcwd())
from log_config import initialize_logging

initialize_logging()


import argparse
import torch
import os
import json
import shortuuid
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

import logging

from marine.utils.utils import get_chunk, get_answers_file_name, get_model_name_from_path
from marine.utils.utils_dataset import *
from marine.utils.utils_guidance import GuidanceLogits
from marine.utils.utils_model import load_model


# -----------------------------------------------------------------------------------------------
# General Configurations
# -----------------------------------------------------------------------------------------------
import mlflow
import pymongo
import requests
from s3fs import S3FileSystem
from log_config import initialize_logging
initialize_logging()
S3_ENDPOINT_URL       = os.environ["S3_ENDPOINT_URL"]
S3_ACCESS_KEY_ID      = os.environ["S3_ACCESS_KEY_ID"]
S3_SECRET_ACCESS_KEY  = os.environ["S3_SECRET_ACCESS_KEY"]
MONGO_ENDPOINT        = os.environ["MONGO_ENDPOINT"]
MONGO_PORT            = os.environ["MONGO_PORT"]
MONGO_USERNAME        = os.environ["MONGO_USERNAME"]
MONGO_PASSWORD        = os.environ["MONGO_PASSWORD"]
N8N_ENDPOINT_URL      = os.environ["N8N_ENDPOINT_URL"]
N8N_WEBHOOK_ID        = os.environ["N8N_WEBHOOK_ID"]

mongo = pymongo.MongoClient(
    host=MONGO_ENDPOINT,
    port=int(MONGO_PORT),
    username=MONGO_USERNAME,
    password=MONGO_PASSWORD,
)
s3 = S3FileSystem(
    anon=False, 
    endpoint_url=S3_ENDPOINT_URL,
    key=S3_ACCESS_KEY_ID,
    secret=S3_SECRET_ACCESS_KEY,
    use_ssl=False
)
storage_options = {
    'key': S3_ACCESS_KEY_ID,
    'secret': S3_SECRET_ACCESS_KEY,
    'endpoint_url': S3_ENDPOINT_URL,
}


    


def report_message_to_n8n(message: str, msg_type: str = "info"):
    try:
        response = requests.post(
            f"{N8N_ENDPOINT_URL}/{N8N_WEBHOOK_ID}",
            json={"message": message, "type": msg_type}
        )
        if response.status_code == 200:
            logging.info("Successfully reported message to n8n.")
        else:
            logging.error(f"Failed to report message to n8n. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Exception occurred while reporting message to n8n: {e}")

def put_file_to_s3(local_path: str, s3_path: str):
    try:
        s3.put(local_path, s3_path)
        logging.info(f"Successfully uploaded `{local_path}` to `{s3_path}`.")
        report_message_to_n8n(f"Successfully uploaded `{local_path}` to `{s3_path}`.")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to {s3_path}. Exception: {e}")
        report_message_to_n8n(f"Failed ro uploaded `{local_path}` to `{s3_path}`.",msg_type="error")


# if "test" in N8N_WEBHOOK_ID:
#     logging.warning("Debug webhook detected.")
#     report_message_to_n8n("Test!", msg_type="info")
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def eval_model(args):

    logging.info(f"Evaluating model: {str(args.model_path)}")
    logging.info(f"Loading questions from {str(args.question_path)} and {str(args.question_file)}...")
    logging.info(f"Loading images from {str(args.image_folder)}...")
    logging.info(f"Using conv mode: {args.conv_mode}")
    logging.info(f"Using num_chunks: {args.num_chunks} and chunk_idx: {args.chunk_idx}")
    logging.info(f"Using temperature: {args.temperature}, top_p: {args.top_p}, max_new_tokens: {args.max_new_tokens}")
    logging.info(f"Using seed: {args.seed}, guidance_strength: {args.guidance_strength}, batch_size: {args.batch_size}, sampling: {args.sampling}")


    print()
    print()
    logging.info("==============================================================================================================================")
    # Model
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    model, tokenizer, processor = load_model(model_name, model_path)

    # QA Data
    questions = json.load(open(os.path.expanduser(
        os.path.join(args.question_path, args.question_file)), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if args.answers_file is None:
        args.answers_file = get_answers_file_name(args, model_name)

    answers_file = os.path.expanduser(
        os.path.join(args.answer_path, args.answers_file))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = COCOEvalDataset(
        questions, 
        args.image_folder,
        processor, tokenizer, 
        args.conv_mode, 
        getattr(model.config, 'mm_use_im_start_end', False)
        custom_flavor='instructblip'
    )

    collator = Collator(processor, model.device)
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.dict_collate_fn)

    report_message_to_n8n(f"Model loaded. Starting evaluation on {len(questions)} questions...", msg_type="info")

    # generate
    sample_out = (None, None)
    for data_batch in tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
        # -----------------------------------------------------------------------------------------------
        # Data preparation
        # -----------------------------------------------------------------------------------------------
        prompts = data_batch['prompts']
        question_ids = data_batch['question_ids']
        img_ids = data_batch['img_ids']
        full_prompts = data_batch['full_prompts']
        guidance_prompts = data_batch['full_prompts_neg']
        global_input_images = data_batch['global_input_images']

        inputs = processor(
            images = global_input_images,
            text = full_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        guidance_inputs = processor(
            images = global_input_images,
            text = guidance_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)


        # -----------------------------------------------------------------------------------------------
        # Inference
        # -----------------------------------------------------------------------------------------------
        max_length = inputs['input_ids'].shape[1] + args.max_new_tokens
        with torch.inference_mode():
            if args.guidance_strength == 0:
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    max_length=max_length,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
            else:
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_length=max_length,
                    min_length=1,
                    num_return_sequences=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                    logits_processor=LogitsProcessorList([
                        GuidanceLogits(
                            guidance_strength=args.guidance_strength,
                            guidance_inputs=guidance_inputs,
                            model=model,
                            tokenizer=tokenizer
                        ),
                    ])
                )
        input_token_len = inputs["input_ids"].shape[1]

        # -----------------------------------------------------------------------------------------------
        # Batch decode the outputs
        # -----------------------------------------------------------------------------------------------
        decoded_outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)

        for i, output in tqdm(enumerate(decoded_outputs), total=len(decoded_outputs), desc="Processing outputs"):
            # Process each output
            output = output.strip()
            logging.info(f"{question_ids[i]}: {output}")

            # Generate answer ID and write to file
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": question_ids[i],
                                       "image_id": img_ids[i],
                                       "prompt": prompts[i],
                                       "text": output,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            
            sample_out = (question_ids[i], {output})
        ans_file.flush()

    ans_file.close()



    put_file_to_s3(answers_file, f"s3://results/CAP6614_MARINE/answers/instructblip/{os.path.basename(answers_file)}")

    report_message_to_n8n(f"Sample output: `{sample_out[0]}` -> `{sample_out[1]}`", msg_type="info")
    report_message_to_n8n(f"InstructBLIP evaluation done! Answers saved to `{answers_file}`", msg_type="info")
    logging.info(f"Done! Saved answers to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Salesforce/instructblip-vicuna-7b")
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--question_path", type=str, default="./data/marine_qa/question")
    parser.add_argument("--question_file", type=str, default="I02_mmc4_grey_ram_th0.68_detr_th0.95.json")
    parser.add_argument("--answer_path", type=str, default="./data/marine_qa/answer")
    parser.add_argument("--answers_file", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_strength", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    try:
        report_message_to_n8n(f"Starting InstructBLIP inference with guidance strength {args.guidance_strength} on questions from ```{args.question_file}``` to generate answers to ```{args.answer_path}```.")
        eval_model(args)
    except Exception as e:
        logging.error(f"Error during run: {e}. Traceback: {traceback.format_exc()}")

        error_log_collection = mongo['LOGS']['CAP6614_MARINE'].insert_one({
            'timestamp': datetime.now(),
            'model_path': args.model_path,
            'question_file': args.question_file,
            'answers_file': args.answers_file,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'model_name': get_model_name_from_path(args.model_path),
        })
        logging.info(f"Error logged with id: {error_log_collection.inserted_id}")
        report_message_to_n8n(
f"""Exception occurred during InstructBLIP evaluation. TL;DR: 
```{str(e)}```.
Traceback: 
```
{traceback.format_exc()}
```
""", msg_type="error")


