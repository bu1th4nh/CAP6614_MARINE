import os
import sys
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
        # custom_flavor='instructblip'
    )

    collator = Collator(processor, model.device)
    eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.dict_collate_fn_with_process)

    # generate
    for data_batch in tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):

        prompts = data_batch["prompts"]
        img_ids = data_batch["img_ids"]
        question_ids = data_batch["question_ids"]
        inputs = data_batch["inputs"]
        guidance_inputs = data_batch["guidance_inputs"]

        with torch.inference_mode():
            if args.guidance_strength == 0:
                output_ids = model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
            else:
                output_ids = model.generate(
                    **inputs,
                    num_beams=5,
                    max_new_tokens=256,
                    min_length=1,
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

        # Batch decode the outputs
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

        ans_file.flush()
    ans_file.close()
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
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_strength", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    eval_model(args)