import json
from fastapi import HTTPException
from typing import Any, Optional, List

from extensions.llm_server.prompt import PROMPTS
from extensions.llm_server.clients import OpenAIClient, AnthropicClient, GeminiClient, MODEL_REGISTRY
from extensions.llm_server.models import ChatMessage


# Helper function to parse model identifier
def parse_model_identifier(model: str) -> tuple[str, str]:
    """Parse model identifier into provider and model name"""
    if '/' not in model:
        raise ValueError("Model must include provider prefix (e.g., 'openai/gpt-4')")
    
    provider, model_name = model.split('/', 1)
    
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}")
    
    if model_name not in MODEL_REGISTRY[provider]:
        raise ValueError(f"Unsupported model: {model_name} for provider: {provider}")
    
    return provider, model_name



# Client Factory
def create_client(provider: str, api_key: str, base_url: Optional[str] = None):
    """Create appropriate client based on provider"""
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key required for {provider}")
    
    if provider == "openai":
        return OpenAIClient(api_key, base_url)
    elif provider == "anthropic":
        return AnthropicClient(api_key)
    elif provider == "gemini":
        return GeminiClient(api_key)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


async def get_mcp_tools(provider: str, mcp_client):
    async with mcp_client:
        mtools = await mcp_client.list_tools()
        if provider == "openai":
            return [{
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema
            } for t in mtools]
        elif provider == "anthropic":
            return [{
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema
            } for t in mtools]
        elif provider == "gemini":
            return [{
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema
            } for t in mtools]
    return []


async def execute_mcp_function(client, function_to_call):
    """
    Execute MCP function using the client based on function call information from LLM
    """
    function_name = function_to_call.get("function_name")
    function_args = function_to_call.get("function_args", {})
    
    print(f"Executing function: {function_name}")
    print(f"Arguments: {function_args}")
    
    try:
        # Call the MCP tool using client.call_tool
        async with client:
            result = await client.call_tool(function_name, function_args)
            print(f"Function executed successfully")
            return result
        
    except Exception as e:
        error_msg = f"⚠️ Error executing {function_name}: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    

def serialize_response(response):
    """
    Extracts and parses the first `text` field from a list of TextContent-like objects.
    If it's not valid JSON, returns the raw text.
    """
    if isinstance(response, list) and len(response) > 0:
        text = getattr(response[0], 'text', None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw_text": text}
    return {"error": "Invalid response format or empty response"}


def format_messages_with_function_call(provider: str, messages: List[ChatMessage], function_to_call: dict, function_response: Any) -> List[ChatMessage]:
    """
    Format messages with function call and response based on provider requirements
    """
    updated_messages = messages.copy()

    prompt = None
    if function_to_call['function_name'] == "search_documents":
        if 'mode' in function_to_call['function_args'].keys():
            if function_to_call['function_args']['mode'] == 'naive':
                prompt = PROMPTS['naive_rag_response']
        else:
            prompt = PROMPTS['rag_response']

    else:
        prompt = "Provide your answer."

    
    
    if provider == "openai":
        # OpenAI format: assistant message with tool_calls, then tool message with response
        updated_messages.append(ChatMessage(
            role="assistant",
            content=f"I'll call the {function_to_call['function_name']} function to help you."
        ))
        
        # Add tool response as a user message with context
        function_result = serialize_response(function_response)
        tool_response_content = f"Function {function_to_call['function_name']} returned: {json.dumps(function_result, indent=2)}"
        
        updated_messages.append(ChatMessage(
            role="user", 
            content=f"Based on this function result, please provide your response: {tool_response_content}. {prompt}"
        ))
        
    elif provider == "anthropic":
        # Anthropic format: Add tool use and tool result as separate messages
        function_result = serialize_response(function_response)
        
        # Add assistant message indicating tool use
        updated_messages.append(ChatMessage(
            role="assistant",
            content=f"I'll use the {function_to_call['function_name']} function to help you."
        ))
        
        # Add tool result as user message with context
        tool_response_content = f"Tool result from {function_to_call['function_name']}: {json.dumps(function_result, indent=2)}"
        updated_messages.append(ChatMessage(
            role="user",
            content=f"Based on this tool result, please provide your final response: {tool_response_content}. {prompt}"
        ))
        
    elif provider == "gemini":
        # Gemini format: Add function call and response in a conversational manner
        function_result = serialize_response(function_response)
        
        # Add function call context
        updated_messages.append(ChatMessage(
            role="assistant",
            content=f"I called the {function_to_call['function_name']} function with arguments: {json.dumps(function_to_call.get('function_args', {}))}"
        ))
        
        # Add function response
        tool_response_content = f"Function result: {json.dumps(function_result, indent=2)}"
        updated_messages.append(ChatMessage(
            role="user",
            content=f"Here's the function result: {tool_response_content}. {prompt}"
        ))
    
    return updated_messages
    