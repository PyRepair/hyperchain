# HyperChain

**HyperChain** is an easy-to-use and efficient Python library that simplifies interacting with various Large Language Models. It allows for asynchronous execution and chaining of the LLMs using customizable prompt templates.

# Installation
To install HyperChain directly from this GitHub repository, execute the following commands:
```bash
git clone https://github.com/PyRepair/hyperchain.git
cd hyperchain
pip install .
```

# Usage
Below are some simple examples to get you started with HyperChain. More detailed examples can be found in the [examples folder](./examples/).
## Prompt Templates
HyperChain offers templates to create prompts easily for different LLM applications. These templates can be combined using the '+' operator.

### StringTemplate
The **StringTemplate** is mainly used with completion models and takes a formatable string as input. In most cases, using a python string is compatible in place of a StringTemplate as it relies on the ```format``` method.
```python
from hyperchain.prompt_templates import StringTemplate

template = StringTemplate("Answer the following question: {question}\n")
```

### ChatTemplate
The **ChatTemplate** is used with chat models and takes a list of formatable messages as input. This is illustrated with OpenAI's chat models in [example_chat_chain.py](./examples/example_chat_chain.py).
```python
from hyperchain.prompt_templates import ChatTemplate

template = ChatTemplate(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{question}"},
    ],
)
```

### MaskToSentinelTemplate
The **MaskToSentinelTemplate** simplifies converting a masked prompt into one that uses sentinel tokens, necessary for some Seq2Seq models like T5. This avoids manual tracking of sentinel tokens and their order, as demonstrated with the CodeT5 model in [example_masked_code.py](./examples/example_masked_code.py).
```python
from hyperchain.prompt_templates import MaskToSentinelTemplate

template = MaskToSentinelTemplate("{masked_code}")

template.format(masked_code="def greet_user(<mask>: User):\n  print('Hi,' + <mask>)\n")
```

## LLMRunner
An **LLMRunner** is used to communicate with a specific LLM inside a chain and provide it with error handlers for different LLM-specific exceptions.
```python
from hyperchain.llm_runners import OpenAIRunner

llm_runner = OpenAIRunner(
    model="MODEL",
    api_key="OPENAI_API_KEY",
    model_params={"max_tokens": 500},
)
```
## LLMChain

**LLMChain** allows using templates to create prompts and send them to a chosen Large-Language Model. It supports asynchronous execution and includes error handling for exceptions identified in each LLMRunner.
```python
from hyperchain.llm_chain import LLMChain

chain = LLMChain(
   template=template,
   llm_runner=runner,
   output_name="answer",
)
```
The **output_name** argument allows the output from one chain to be used as the input for the next, under a specified name as seen in [example_chain.py](./examples/example_chain.py)