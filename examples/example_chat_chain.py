from hyperchain.prompt_templates import ChatTemplate
from hyperchain.llm_chain import LLMChain
from hyperchain.llm_model_runners import OpenAIChatRunner

chat_template = ChatTemplate(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{question}"},
    ],
)

chat_template_2 = chat_template + [
    {"role": "system", "content": "{answer}"},
    {"role": "user", "content": "{question2}"},
]

chat_template_result = chat_template_2 + [
    {"role": "user", "content": "{answer2}"}
]

chat_runner = OpenAIChatRunner(
    api_key="ENTER API KEY HERE OR IN ENV VARIABLE",
    model_params={"max_tokens": 600},
)

llm_chain = LLMChain(
    template=chat_template, llm_runner=chat_runner, output_name="answer"
) + LLMChain(
    template=chat_template_2,
    llm_runner=chat_runner,
)

response = llm_chain.run(
    question="How are you?", question2="What can you assist me with?"
)

print(
    chat_template_result.format(
        question="How are you?",
        answer=response.previous_result.output,
        question2="What can you assist me with?",
        answer2=response.output,
    )
)
