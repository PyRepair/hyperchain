from hyperchain.prompt_templates import StringTemplate
from hyperchain.llm_chain import LLMChain
from hyperchain.llm_runners import OpenAIRunner

template_prepare = StringTemplate(
    "Answer the following question:\n{question}\n\nShort answer is:\n"
)
template_guess = StringTemplate(
    "Guess what question this was an answer to:\n{answer}\n\nGuessed question:\n"
)

llm_chain_prepare = LLMChain(
    template=template_prepare,
    llm_runner=OpenAIRunner(
        api_key="ENTER API KEY HERE OR IN ENV VARIABLE",
        model_params={"max_tokens": 600},
    ),
    output_name="answer",
)

llm_chain_guess = LLMChain(
    template=template_guess,
    llm_runner=OpenAIRunner(
        api_key="ENTER API KEY HERE OR IN ENV VARIABLE",
        model_params={"max_tokens": 50},
    ),
)

llm_chain = llm_chain_prepare + llm_chain_guess

print(llm_chain.run(question="What is APR?").output)  # Run chain of querries
