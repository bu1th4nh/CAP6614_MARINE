import os
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import logging
from copy import deepcopy

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

from typing import Iterable
from transformers import BatchEncoding


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
        mm_use_im_start_end: bool,
        custom_flavor: str = None
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
        image = Image.open(image_path)

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

        # Tokenize
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        guidance_inputs = self.processor(text=full_prompt_neg, images=image, return_tensors="pt")
        
        if self.custom_flavor == "instructblip":
            return (
                cur_prompt,
                question_id,
                img_id,
                inputs,
                guidance_inputs
            )
        else:
            return (
                cur_prompt,
                question_id,
                img_id,
                inputs["input_ids"],
                guidance_inputs["input_ids"],
                inputs["pixel_values"],
                guidance_inputs["pixel_values"],
                inputs["attention_mask"],
                guidance_inputs["attention_mask"]
            )


def _pad_left_1d(seqs: Iterable[torch.Tensor], pad_value: int) -> torch.Tensor:
    # seqs: list of 1D tensors [L_i]
    flipped = [s.flip(0) for s in seqs]
    out = pad_sequence(flipped, batch_first=True, padding_value=pad_value).flip(1)
    return out  # (B, Lmax), left-padded

def _stack_or_pad_enc_list(enc_list: Tuple[Any, ...],
                           pad_token_id: int = 0,
                           left_pad: bool = True) -> BatchEncoding:
    """
    Merge a tuple of per-item encodings (BatchEncoding or dict-like) into one batched BatchEncoding.
    - 1D sequences -> pad
    - 3D tensors   -> stack (images: C,H,W)
    - If tensors arrive with a fake leading batch dim (1, ...), squeeze it.
    """
    # Normalize each item to a dict of tensors
    norm = []
    for e in enc_list:
        if isinstance(e, BatchEncoding):
            d = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in e.items()}
        elif isinstance(e, dict):
            d = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in e.items()}
        else:
            raise TypeError(f"Unsupported encoding type: {type(e)}")
        # squeeze fake leading batch dim some processors return
        for k, v in d.items():
            if v.ndim >= 2 and v.shape[0] == 1:
                d[k] = v.squeeze(0)
        norm.append(d)

    keys = set().union(*(d.keys() for d in norm))
    merged = {}

    for k in keys:
        vals = [d[k] for d in norm if k in d]
        if not vals:
            continue
        d0 = vals[0].ndim
        if d0 == 1:
            merged[k] = _pad_left_1d(vals, pad_value=pad_token_id) if left_pad \
                        else pad_sequence(vals, batch_first=True, padding_value=pad_token_id)
        elif d0 == 3:
            # image-like (C,H,W)
            merged[k] = torch.stack(vals, dim=0)  # (B, C, H, W)
        else:
            # Fallback: try stack (e.g., 2D masks HxW or already-batched extras)
            merged[k] = torch.stack(vals, dim=0)

    return BatchEncoding(merged)

# ---------- the collate you asked for ----------

def make_unified_collate(tokenizer_pad_id: int = 0, left_pad: bool = True):
    """
    Always returns:
        prompts, question_ids, img_ids, inputs, guidance_inputs
    - inputs/guidance_inputs are BatchEncoding with padded ids/masks and stacked pixel_values.
    Supports both:
      (cur_prompt, question_id, img_id, BatchEncoding, BatchEncoding)          # instructblip path
      (cur_prompt, question_id, img_id, ids, gids, pix, gpix, mask, gmask)     # llava-style path
    """
    def _collate(batch: List[Tuple[Any, ...]]):
        first = batch[0]
        # InstructBLIP tuple length == 5
        if len(first) == 5:
            prompts, qids, img_ids, inputs, guidance_inputs = zip(*batch)
            fin_inputs   = _stack_or_pad_enc_list(inputs,   pad_token_id=tokenizer_pad_id, left_pad=left_pad)
            fin_guidance = _stack_or_pad_enc_list(guidance_inputs, pad_token_id=tokenizer_pad_id, left_pad=left_pad)
            return list(prompts), list(qids), list(img_ids), fin_inputs, fin_guidance

        # LLaVA-style tuple length == 9
        elif len(first) == 9:
            (
                prompts, qids, img_ids,
                input_ids_list, guidance_ids_list,
                image_tensors, guidance_image_tensors,
                attention_masks_list, guidance_attention_masks_list
            ) = zip(*batch)

            # normalize image shapes to (C,H,W)
            def _s3(x: torch.Tensor) -> torch.Tensor:
                return x if x.ndim == 3 else x.squeeze(0)

            inputs_enc = {
                "input_ids":       _pad_left_1d([t.squeeze(0) for t in input_ids_list], pad_value=tokenizer_pad_id),
                "attention_mask":  _pad_left_1d([t.squeeze(0) for t in attention_masks_list], pad_value=0),
                "pixel_values":    torch.stack([_s3(t) for t in image_tensors], dim=0),
            }
            guidance_enc = {
                "input_ids":       _pad_left_1d([t.squeeze(0) for t in guidance_ids_list], pad_value=tokenizer_pad_id),
                "attention_mask":  _pad_left_1d([t.squeeze(0) for t in guidance_attention_masks_list], pad_value=0),
                "pixel_values":    torch.stack([_s3(t) for t in guidance_image_tensors], dim=0),
            }

            fin_inputs   = BatchEncoding(inputs_enc)
            fin_guidance = BatchEncoding(guidance_enc)
            return list(prompts), list(qids), list(img_ids), fin_inputs, fin_guidance

        else:
            raise ValueError(f"Unexpected sample tuple size {len(first)}; expected 5 or 9.")
    return _collate
