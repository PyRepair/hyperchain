import pytest
from hyperchain.chain.function_chain import FunctionChain

@pytest.mark.asyncio
async def test_async_run_async_function():
    async def async_function(inputs):
        return {"result": "async_result"}
    
    function_chain = FunctionChain(async_function)
    result = await function_chain.async_run()
    assert result.result == "async_result"

@pytest.mark.asyncio
async def test_async_run_sync_function():
    def sync_function(inputs):
        return {"result": "synchronous_result"}
    
    function_chain = FunctionChain(sync_function)
    result = await function_chain.async_run()
    assert result.result == "synchronous_result"

def test_sync_run_async_function():
    async def async_function(inputs):
        return {"result": "async_result"}
    
    function_chain = FunctionChain(async_function)
    result = function_chain.run()
    assert result.result == "async_result"

def test_sync_run_sync_function():
    def sync_function(inputs):
        return {"result": "synchronous_result"}
    
    function_chain = FunctionChain(sync_function)
    result = function_chain.run()
    assert result.result == "synchronous_result"