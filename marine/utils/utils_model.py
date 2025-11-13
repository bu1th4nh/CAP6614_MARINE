# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: utils_model.py
# Date: 2025/11/12 13:53:56
# Description: 
# 
# (c) 2025 bu1th4nh. All rights reserved. 
# Written with dedication at the University of Central Florida, EPCOT, and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------

import torch

def load_model(model_name: str, model_path: str):
    """
    Load vision-language models and associated components based on model name.
    
    Args:
        model_name (str): Name of the model ('llava2', 'mplug_owl2', etc.)
        model_path (str): Path or hub name of the pretrained model.

    Returns:
        A dictionary containing loaded components: model, tokenizer, processor/image_processor
    """
    model_name = model_name.lower()
    if "llava" in model_name:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(model_path).cuda()
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer

        return model, tokenizer, processor
    

    elif model_name == "mplug-owl3":
        from modelscope import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)

        model.eval().cuda()
        return model, tokenizer, processor
    

    elif "instructblip" in model_name:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path).cuda()
        processor = InstructBlipProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer


        # Make sure processor carries the model’s num_query_tokens
        if getattr(processor, "num_query_tokens", None) is None:
            processor.num_query_tokens = model.config.num_query_tokens  # important post-v4.46

        return model, tokenizer, processor
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
