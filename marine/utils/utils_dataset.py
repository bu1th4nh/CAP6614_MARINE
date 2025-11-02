import os
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from PIL import Image
import logging

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from transformers import BatchEncoding
from typing import Iterable, Mapping, Any, List, Tuple

try:
    from transformers.image_processing_utils import BatchFeature
except Exception:
    # fallback in case of older/newer HF; treat as Mapping
    class BatchFeature(dict):
        pass



class COCOEvalDataset(Dataset):
    """
    Dataset class for COCO-style evaluation of LVLMs.
    Each item includes the image, prompt, negative prompt, and tokenized inputs.
    """
    def __init__(
        self,
        questions: List[dict],
        image_dir: str,
        processor,
        tokenizer,
        conv_mode: str,
        mm_use_im_start_end: bool = False,
        custom_flavor: str = None,
    ):
        self.questions = questions
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end
        self.custom_flavor = custom_flavor

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        data = self.questions[idx]
        question_id = data["id"]
        img_id = data.get("image")

        if img_id is None:
            raise ValueError(f"Missing image in question {question_id}")

        image_path = os.path.join(self.image_dir, img_id)
        image = Image.open(image_path).convert("RGB")

        qs = data["conversations"][0]["value"].replace("<image>", "").strip()
        qs_neg = data["conversations"][-1]["value"]

        
        # Add image tokens
        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        prompt = image_token + "\n" + qs
        guidance_prompt = image_token + "\n" + qs_neg
        cur_prompt = "<image>\n" + qs

        # Build conversation prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        conv_neg = conv_templates[self.conv_mode].copy()
        conv_neg.append_message(conv_neg.roles[0], guidance_prompt)
        conv_neg.append_message(conv_neg.roles[1], None)
        full_prompt_neg = conv_neg.get_prompt()

        # logging.fatal(f"Full prompt: {full_prompt}")
        # logging.fatal(f"Full negative prompt: {full_prompt_neg}")

        # Tokenize
        # inputs = self.processor(text=full_prompt, images=image, return_tensors="pt", padding=True, truncation=False)
        # guidance_inputs = self.processor(text=full_prompt_neg, images=image, return_tensors="pt", padding=True, truncation=False)

    
        # return (
        #     cur_prompt,
        #     question_id,
        #     img_id,
        #     inputs,
        #     guidance_inputs
        # )
    
        
        return {
            "cur_prompt": cur_prompt,
            "question_id": question_id,
            "img_id": img_id,
            "full_prompt": full_prompt,
            "full_prompt_neg": full_prompt_neg,
            "image": image,
        }



def dict_collate_fn(batch: List[Mapping[str, Any]]):
    """
    Collate function to batch a list of dictionaries.
    """
    return batch


def custom_collate_fn(batch: List[Tuple[
    str,          # cur_prompt
    str,          # question_id
    str,          # img_id
    Any,          # inputs
    Any           # guidance_inputs
]]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function to pad input/guidance_ids and attention masks,
    and stack image tensors. All outputs are moved to CUDA.
    """
    prompts = [item[0] for item in batch]
    question_ids = [item[1] for item in batch]
    img_ids = [item[2] for item in batch]
    inputs_list = [item[3] for item in batch]
    guidance_inputs_list = [item[4] for item in batch]


    # logging.fatal(f"Type of prompts: {type(prompts)}")
    # logging.fatal(f"Type of question_ids: {type(question_ids)}")
    # logging.fatal(f"Type of img_ids: {type(img_ids)}")
    # logging.fatal(f"Type of inputs_list: {type(inputs_list)}")
    # logging.fatal(f"Type of inputs_list[0]: {type(inputs_list[0])}")


    logging.fatal(f"input keys: {inputs_list[0]}")


    
    input_ids_list = [inp["input_ids"] for inp in inputs_list]
    image_tensors = [inp["pixel_values"].squeeze(0) for inp in inputs_list]
    attention_masks_list = [inp["attention_mask"] for inp in inputs_list]


    guidance_ids_list = [g_inp["input_ids"] for g_inp in guidance_inputs_list]  
    guidance_image_tensors = [g_inp["pixel_values"].squeeze(0) for g_inp in guidance_inputs_list]
    guidance_attention_masks_list = [g_inp["attention_mask"] for g_inp in guidance_inputs_list]


    def process_sequence(seq_list):
        seq_list = [seq.squeeze(0).flip(dims=[0]) for seq in seq_list]
        return pad_sequence(seq_list, batch_first=True, padding_value=0).flip(dims=[1])

    input_ids_batch = process_sequence(input_ids_list).cuda()
    attn_mask_batch = process_sequence(attention_masks_list).cuda()
    image_tensor_batch = torch.stack(image_tensors).squeeze(1).cuda()
    
    guidance_ids_batch = process_sequence(guidance_ids_list).cuda()
    guidance_attn_mask_batch = process_sequence(guidance_attention_masks_list).cuda()
    guidance_image_tensor_batch = torch.stack(guidance_image_tensors).squeeze(1).cuda()



    rtn_values = {
        "prompts": list(prompts),
        "question_ids": list(question_ids),
        "img_ids": list(img_ids),
        "inputs": {
            "input_ids": input_ids_batch,
            "attention_masks": attn_mask_batch,
            "pixel_values": image_tensor_batch,
        },
        "guidance_inputs": {
            "input_ids": guidance_ids_batch,
            "attention_masks": guidance_attn_mask_batch,
            "pixel_values": guidance_image_tensor_batch,
        }
    }


    if 'qformer_input_ids' in inputs_list[0]:
        qformer_input_ids_list = [inp['qformer_input_ids'] for inp in inputs_list]
        qformer_attention_masks_list = [inp['qformer_attention_mask'] for inp in inputs_list]
        qformer_input_ids_batch = process_sequence(qformer_input_ids_list).cuda()
        qformer_attn_mask_batch = process_sequence(qformer_attention_masks_list).cuda()

        rtn_values['inputs']['qformer_input_ids'] = qformer_input_ids_batch
        rtn_values['inputs']['qformer_attention_masks'] = qformer_attn_mask_batch

    if 'qformer_input_ids' in guidance_inputs_list[0]:
        guidance_qformer_input_ids_list = [inp['qformer_input_ids'] for inp in guidance_inputs_list]
        guidance_qformer_attention_masks_list = [inp['qformer_attention_mask'] for inp in guidance_inputs_list]
        guidance_qformer_input_ids_batch = process_sequence(guidance_qformer_input_ids_list).cuda()
        guidance_qformer_attn_mask_batch = process_sequence(guidance_qformer_attention_masks_list).cuda()

        rtn_values['guidance_inputs']['qformer_input_ids'] = guidance_qformer_input_ids_batch
        rtn_values['guidance_inputs']['qformer_attention_masks'] = guidance_qformer_attn_mask_batch

    return rtn_values