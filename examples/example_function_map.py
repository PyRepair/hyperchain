from hyperchain.chain import LLMChain, FunctionChain, FunctionMapChain
from hyperchain.llm_runners import T5ConditionalModelRunner
from hyperchain.prompt_templates import MaskToSentinelTemplate

import re

async def extract_words(response):
    return [word for word in re.split(r'[^a-zA-Z]', str(response)) if len(word) > 3]

llm_chain = LLMChain(
    template=MaskToSentinelTemplate("{masked_code}"),
    llm_runner=T5ConditionalModelRunner(
        model="Salesforce/codet5p-770m",
        model_kwargs={"max_new_tokens": 100},
    )
)

function_map_chain = FunctionMapChain([
    ("result", extract_words, "word_list")
])

chain_result = (llm_chain + function_map_chain).run(masked_code="def greet_user(<mask>: User):\n  print('Hi,' + <mask>)\n")

print(chain_result.result)
print(chain_result.word_list)