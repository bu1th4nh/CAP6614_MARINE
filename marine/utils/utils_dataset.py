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
        image_token = ""
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        prompt = qs
        guidance_prompt = qs_neg
        cur_prompt = qs

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


class Collator:
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def dict_collate_fn_with_process(self, batch: List[Mapping[str, Any]]):
        # Metadata
        prompts = [x["cur_prompt"] for x in batch]
        question_ids = [x["question_id"] for x in batch]
        img_ids = [x["img_id"] for x in batch]

        # Prepare inputs
        global_input_images = [x["image"].resize((224, 224)) for x in batch]
        input_prompts = [x["full_prompt"].replace("<image>", "") for x in batch]
        guidance_prompts = [x["full_prompt_neg"].replace("<image>", "") for x in batch]

        # input_prompts = ["What's in this image?" for x in batch]
        # guidance_prompts = ["Describe this image in detail." for x in batch]

        
        inputs = self.processor(
            images=global_input_images, 
            text=input_prompts, 
            return_tensors="pt",
            padding=True,
        ).to(self.device)


        guidance_inputs = self.processor(
            images=global_input_images, 
            text=guidance_prompts, 
            return_tensors="pt",
            padding=True,
        ).to(self.device)


        return {
            "prompts": list(prompts),
            "question_ids": list(question_ids),
            "img_ids": list(img_ids),
            "inputs": inputs,
            "guidance_inputs": guidance_inputs
        }
    

    def dict_collate_fn(self, batch: List[Mapping[str, Any]]):
        # Prepare inputs
        prompts = [x["cur_prompt"] for x in batch]
        question_ids = [x["question_id"] for x in batch]
        img_ids = [x["img_id"] for x in batch]
        global_input_images = [x["image"].resize((224, 224)) for x in batch]
        input_prompts = [x["full_prompt"].replace("<image>", "") for x in batch]
        guidance_prompts = [x["full_prompt_neg"].replace("<image>", "") for x in batch]

        return {
            "cur_prompt": prompts,
            "question_id": question_ids,
            "img_id": img_ids,
            "full_prompt": input_prompts,
            "full_prompt_neg": guidance_prompts,
            "image": global_input_images,
        }

    def bypass_collate_fn(self, batch: List[Mapping[str, Any]]):
        return batch


