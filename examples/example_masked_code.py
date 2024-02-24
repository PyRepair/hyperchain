from hyperchain.llm_chain import LLMChain
from hyperchain.llm_model_runners.t5_model_runner import T5ConditionalModelRunner
from hyperchain.llm_model_runners.masked_model_runner import MaskedModelRunner
from hyperchain.prompt_templates import MaskToSentinelTemplate

llm_chain_codet5 = LLMChain(
    template=MaskToSentinelTemplate("{masked_code}"),
    llm_runner=T5ConditionalModelRunner(model="Salesforce/codet5p-770m")
)

llm_chain_codebert = LLMChain(
    template="{masked_code}",
    llm_runner=MaskedModelRunner(model="microsoft/codebert-base-mlm")
)

print(llm_chain_codet5.run(masked_code="def greet_user(<mask>: User):\n  print('Hi,' + <mask>)\n").output)
print(llm_chain_codebert.run(masked_code="def greet_user(<mask>: User):\n  print('Hi,' + <mask>.<mask>)\n").output)