import pytest
from hyperchain.chain.function_map_chain import FunctionMapChain

@pytest.mark.parametrize("function_list, key_list",[
    ([("from", lambda x: x, "to")], ["from"]),
    ([("from", lambda x: x)], ["from"]),
    ([], []),
])
def test_required_keys(function_list, key_list):
    function_map_chain = FunctionMapChain(function_list)
    assert function_map_chain.required_keys == key_list

@pytest.mark.parametrize("function_list, key_list",[
    ([("from", lambda x: x, "to")], ["to"]),
    ([("from", lambda x: x)], ["from"]),
    ([], []),
])
def test_output_keys(function_list, key_list):
    function_map_chain = FunctionMapChain(function_list)
    assert function_map_chain.output_keys == key_list

@pytest.mark.asyncio
async def test_async_run_async_function():
    async def async_function(input_string):
        return input_string + "!"
    
    function_map_chain = FunctionMapChain([("result", async_function, "result_new")])
    result = await function_map_chain.async_run(result="hello")
    assert result.result_new == "hello!"

@pytest.mark.asyncio
async def test_async_run_sync_function():
    def sync_function(input_string):
        return input_string + "!"
    
    function_map_chain = FunctionMapChain([("result", sync_function, "result_new")])
    result = await function_map_chain.async_run(result="hello")
    assert result.result_new == "hello!"

def test_sync_run_async_function():
    async def async_function(input_string):
        return input_string + "!"
    
    function_map_chain = FunctionMapChain([("result", async_function, "result_new")])
    result = function_map_chain.run(result="hello")
    assert result.result_new == "hello!"

def test_sync_run_sync_function():
    def sync_function(input_string):
        return input_string + "!"
    
    function_map_chain = FunctionMapChain([("result", sync_function, "result_new")])
    result = function_map_chain.run(result="hello")
    assert result.result_new == "hello!"