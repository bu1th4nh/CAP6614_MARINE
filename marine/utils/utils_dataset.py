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
    flipped = [s.flip(0) for s in seqs]  # right-pad after flip
    out = pad_sequence(flipped, batch_first=True, padding_value=pad_value).flip(1)
    return out  # (B, Lmax), left-padded

def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        raise TypeError(f"Value cannot be converted to tensor: {type(x)}")

def _normalize_encoding(e: Any) -> dict:
    """
    Turn BatchEncoding / BatchFeature / Mapping into a dict[str, torch.Tensor],
    squeezing any fake leading batch dim of size 1.
    """
    if isinstance(e, (BatchEncoding, BatchFeature, Mapping)):
        items = dict(e.items())
    else:
        raise TypeError(f"Unsupported encoding type in collate: {type(e)}")

    out = {}
    for k, v in items.items():
        t = _to_tensor(v)
        if t.ndim >= 2 and t.shape[0] == 1:
            t = t.squeeze(0)  # many processors return (1, L) or (1, C, H, W)
        out[k] = t
    return out

def _stack_or_pad_enc_list(enc_list: Tuple[Any, ...],
                           pad_token_id: int = 0,
                           left_pad: bool = True) -> BatchEncoding:
    """
    Merge a tuple of per-item encodings into one batched BatchEncoding.
    1-D -> pad; 3-D (C,H,W) -> stack; else try stack.
    """
    norm = [_normalize_encoding(e) for e in enc_list]
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
            # images (C,H,W)
            merged[k] = torch.stack(vals, dim=0)  # (B, C, H, W)
        else:
            # e.g., (H,W) masks or already batched extras -> stack
            merged[k] = torch.stack(vals, dim=0)

    return BatchEncoding(merged)

# ---- unified collate that returns the 5-tuple you asked for ----

def make_unified_collate(tokenizer_pad_id: int = 0, left_pad: bool = True):
    """
    Always returns:
        prompts, question_ids, img_ids, inputs, guidance_inputs

    Supports both dataset flavors:
      (cur_prompt, question_id, img_id, BatchEncoding/BatchFeature, BatchEncoding/BatchFeature)
      (cur_prompt, question_id, img_id, ids, gids, pix, gpix, mask, gmask)
    """
    def _collate(batch: List[Tuple[Any, ...]]):
        first = batch[0]

        if len(first) == 5:
            # InstructBLIP-style: encodings already produced by processor
            prompts, qids, img_ids, inputs, guidance = zip(*batch)
            fin_inputs   = _stack_or_pad_enc_list(inputs,   pad_token_id=tokenizer_pad_id, left_pad=left_pad)
            fin_guidance = _stack_or_pad_enc_list(guidance, pad_token_id=tokenizer_pad_id, left_pad=left_pad)
            return list(prompts), list(qids), list(img_ids), fin_inputs, fin_guidance

        elif len(first) == 9:
            # LLaVA-style: raw tensors
            (
                prompts, qids, img_ids,
                input_ids_list, guidance_ids_list,
                image_tensors, guidance_image_tensors,
                attention_masks_list, guidance_attention_masks_list
            ) = zip(*batch)

            def s3(t: torch.Tensor) -> torch.Tensor:
                return t if t.ndim == 3 else t.squeeze(0)

            fin_inputs = BatchEncoding({
                "input_ids":      _pad_left_1d([t.squeeze(0) for t in input_ids_list], pad_value=tokenizer_pad_id),
                "attention_mask": _pad_left_1d([t.squeeze(0) for t in attention_masks_list], pad_value=0),
                "pixel_values":   torch.stack([s3(t) for t in image_tensors], dim=0),
            })
            fin_guidance = BatchEncoding({
                "input_ids":      _pad_left_1d([t.squeeze(0) for t in guidance_ids_list], pad_value=tokenizer_pad_id),
                "attention_mask": _pad_left_1d([t.squeeze(0) for t in guidance_attention_masks_list], pad_value=0),
                "pixel_values":   torch.stack([s3(t) for t in guidance_image_tensors], dim=0),
            })

            return list(prompts), list(qids), list(img_ids), fin_inputs, fin_guidance

        else:
            raise ValueError(f"Unexpected sample tuple size {len(first)}; expected 5 or 9.")

    return _collate