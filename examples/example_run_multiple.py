from hyperchain.prompt_templates import StringTemplate
from hyperchain.chain import LLMChain
from hyperchain.llm_runners import OpenAIRunner

template = StringTemplate("Question: {question}\nAnswer:\n")

llm_chain = LLMChain(
    template=template,
    llm_runner=OpenAIRunner(
        model="gpt-3.5-turbo-instruct",
        api_key="ENTER API KEY HERE OR IN ENV VARIABLE",
        model_params={"max_tokens": 40},
    ),
)

print(llm_chain.run(question="What is APR?"))  # Run one querry
print(
    llm_chain.run_multiple(
        {"question": "Tell me about APR (Automatic Program Repair)"},
        {"question": "Tell me about python"},
        {"question": "Tell me about machine learning"},
    )
)  # Run multiple at the same time
print(
    llm_chain.run_multiple(
        *list(
            {"question": "Tell me about APR (Automatic Program Repair)"}
            for _ in range(30)
        )
    )
)  # Run multiple unwrapping list
