from .llm_runner import LLMRunner, LLMResult
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline        
import logging
import numpy as np

class MaskedModelRunner(LLMRunner):
    def __init__(
        self,
        model,
        tokenizer = None,
        model_kwargs = {},
        pipeline_parameters = {},
    ):
        if isinstance(model, str):
            self.model = AutoModelForMaskedLM.from_pretrained(model)
        else:
            self.model = model
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            if isinstance(model, str):
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                logging.critical("Please specify tokenizer to use with the model.")
                raise ValueError("Tokenizer not specified.")
        else:
            self.tokenizer = tokenizer

        self.pipeline_parameters = pipeline_parameters
        self.model_kwargs = model_kwargs
        self.fill_mask = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, model_kwargs=self.model_kwargs)

    async def async_run(self, prompt: str):
        predictions = self.fill_mask(prompt, **self.pipeline_parameters)
        response = prompt

        for prediction in predictions:
            response = response.replace('<mask>', prediction[0]['token_str'], 1)

        return LLMResult(response, extra_llm_outputs=predictions)

    def _get_error_handlers(self):
        return []