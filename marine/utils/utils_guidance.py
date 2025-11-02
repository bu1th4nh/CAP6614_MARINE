from transformers import (LogitsProcessor)
import torch.nn.functional as F
import logging

class GuidanceLogits(LogitsProcessor):

    def __init__(self, guidance_strength, guidance_inputs, model, tokenizer):
        """
        Args:
            guidance_strength (float): Strength of the guidance.
            guidance_inputs (dict): A package of inputs for guidance model
            model: The vision-language model to be used for guidance.
        """
        self.guidance_strength = guidance_strength
        self.full_guidance_inputs = guidance_inputs

        
        self.guidance = guidance_inputs["input_ids"]
        self.images = guidance_inputs["pixel_values"]
        self.attention_mask = guidance_inputs["attention_mask"]
        self.tokenizer = tokenizer
        self.model = model
        self.out = None
        

        if 'qformer_input_ids' in guidance_inputs:
            self.qformer_input_ids = guidance_inputs["qformer_input_ids"]
            self.qformer_attention_mask = guidance_inputs["qformer_attention_mask"]
        else:
            self.qformer_input_ids = None
            self.qformer_attention_mask = None



    def __call__(self, input_ids, logits):
        logits = F.log_softmax(logits, dim=-1)
        if self.out is None:
            self.out = self.model(**self.full_guidance_inputs, use_cache=True)
        else:
            if self.qformer_input_ids is not None and self.qformer_attention_mask is not None:
                self.out = self.model(
                    input_ids = input_ids[:, -1:],
                    pixel_values=self.images,
                    attention_mask=self.attention_mask,
                    qformer_input_ids=self.qformer_input_ids,
                    qformer_attention_mask=self.qformer_attention_mask,
                    use_cache=True,
                )
            else:
                self.out = self.model(
                    input_ids = input_ids[:, -1:],
                    pixel_values=self.images,
                    attention_mask=self.attention_mask,
                    use_cache=True,
                )

        if len(self.out.logits) == 1:
            guidance_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        else:
            guidance_logits = F.log_softmax(self.out.logits[:,-1:], dim=-1).to(logits.device)
            guidance_logits = guidance_logits.squeeze(1)

        logging.fatal(f"Guidance logits shape: {guidance_logits.shape}, Logits shape: {logits.shape}")
        # Expand guidance_logits to match logits shape
        if guidance_logits.shape != logits.shape:
            guidance_logits = guidance_logits.expand_as(logits)

        

        out = self.guidance_strength * (guidance_logits - logits) + logits
        out = F.log_softmax(out, dim=-1)



        return out