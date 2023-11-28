from hyperchain.prompt_templates import StringTemplate
from hyperchain.llm_chain import LLMChain
from hyperchain.llm_model_runners.hf_runner import HuggingFaceRunner
from hyperchain.llm_model_runners.openai_runner import OpenAIRunner

template_hf = StringTemplate("Answer the following question: {question}\n")

llm_chain_hf = LLMChain(
    template=template_hf,
    llm_runner=HuggingFaceRunner(
        model="gpt2",
        api_key="Hugging Face Hub API KEY",
    ),
    output_name="answer",
)

template_openai = StringTemplate(
    "Guess what question this was an answer to:\n{answer}\n\nGuessed question is:\n"
)

llm_chain_openai = LLMChain(
    template=template_openai,
    llm_runner=OpenAIRunner(
        api_key="OpenAI API KEY",
        model_params={"max_tokens": 50},
    ),
)

combined_chain = llm_chain_hf + llm_chain_openai

print(combined_chain.run(question="How are you doing?").output)  # Run chain
