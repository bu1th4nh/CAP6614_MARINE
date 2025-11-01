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

    elif "instructblip" in model_name:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

        model = InstructBlipForConditionalGeneration.from_pretrained(model_path).cuda()
        processor = InstructBlipProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer

        return model, tokenizer, processor
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
