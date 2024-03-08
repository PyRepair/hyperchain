from .llm_runner import LLMRunner, LLMResult
from transformers import T5ForConditionalGeneration, AutoTokenizer
import logging


class T5ConditionalModelRunner(LLMRunner):
    def __init__(
        self,
        model,
        tokenizer = None,
        model_kwargs = {},
    ):  
        if isinstance(model, str):
            self.model = T5ForConditionalGeneration.from_pretrained(model)
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

        self.model_kwargs = model_kwargs
        self.sentinel_tokens_set = set(self.tokenizer.convert_tokens_to_ids([token for token in self.tokenizer.additional_special_tokens if "extra_id" in token]))

    def _apply_response(self, prompt, response):
        result = []

        for masked_input, generated in zip(prompt.tolist(), response.tolist()):
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(generated, already_has_special_tokens=True)
            sentinel_dict = {}
            sentinels_found = [i for i in range(len(generated)) if generated[i] in self.sentinel_tokens_set]
            for sentinel_idx in sentinels_found:
                sentinel = generated[sentinel_idx]
                if sentinel in sentinel_dict:
                    continue
                
                sentinel_dict[sentinel] = []
                for i in range(sentinel_idx + 1, len(generated)):
                    if special_tokens_mask[i]:
                        break
                    
                    sentinel_dict[sentinel].append(generated[i])
            
            result_curr = []
            for token in masked_input:
                if token in sentinel_dict:
                    result_curr.extend(sentinel_dict[token])
                else:
                    result_curr.append(token)

            result.append(result_curr)
        
        return result

    async def async_run(self, prompt: str):
        from torch import inference_mode
        input_ids  = self.tokenizer(prompt, return_tensors="pt").input_ids
        with inference_mode():
            response = self.model.generate(input_ids, **self.model_kwargs)
        decoded_response = self.tokenizer.decode(self._apply_response(input_ids, response)[0], skip_special_tokens=True)
        return LLMResult(decoded_response, extra_llm_outputs={"input_ids": input_ids, "response": response})
    
    async def run_batch(self, prompts):
        from torch import inference_mode
        input_ids_batch  = self.tokenizer(prompts, padding=True, return_tensors="pt").input_ids
        with inference_mode():
            responses = self.model.generate(input_ids_batch, **self.model_kwargs)
        decoded_responses = self.tokenizer.batch_decode(self._apply_response(input_ids_batch, responses), skip_special_tokens=True)
        return [
            LLMResult(decoded_response, extra_llm_outputs={"input_ids": input_ids, "response": response})
            for decoded_response, input_ids, response in zip(decoded_responses, input_ids_batch, responses)
        ]


    def _get_error_handlers(self):
        return []