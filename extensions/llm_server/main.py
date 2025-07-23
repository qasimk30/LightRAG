import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types
import uvicorn
import logging
from datetime import datetime
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


from extensions.llm_server.clients import MODEL_REGISTRY
from extensions.llm_server.models import ModelProvider, ChatMessage, OpenWebUIRequest, ModelInfo
from extensions.llm_server.utils import create_client, parse_model_identifier, get_mcp_tools, execute_mcp_function, format_messages_with_function_call


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI Application
app = FastAPI(
    title="OpenWebUI Multi-Provider LLM Server",
    description="A unified API server for OpenWebUI supporting OpenAI, Anthropic, and Gemini models with user-provided API keys",
    version="2.0.0"
)

# Add CORS middleware for OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your OpenWebUI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "OpenWebUI Multi-Provider LLM Server",
        "version": "2.0.0",
        "supported_providers": list(MODEL_REGISTRY.keys()),
        "total_models": sum(len(models) for models in MODEL_REGISTRY.values())
    }

@app.get("/v1/models")
async def list_models():
    """List all available models in OpenAI format for OpenWebUI"""
    models = []
    
    for provider, provider_models in MODEL_REGISTRY.items():
        for model_id, model_info in provider_models.items():
            models.append({
                "id": f"{provider}/{model_id}",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": provider,
                "provider": provider,
                "name": model_info["name"],
                "description": model_info["description"]
            })
    
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenWebUIRequest,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """
    Chat completions endpoint compatible with OpenWebUI
    Supports API key from multiple sources:
    1. Request body (request.api_key)
    2. Authorization header (Bearer token)
    3. X-API-Key header
    """
    # print(request)
    # print(request_header.headers)
    # user_name = request_header.headers.get("X-OpenWebUI-User-Name")
    # user_email = request_header.headers.get("X-OpenWebUI-User-Email")
    # user_id = request_header.headers.get("X-OpenWebUI-User-Id")
    # user_role = request_header.headers.get("X-OpenWebUI-User-Role")

    # print(user_name, user_email, user_id, user_role)

    # Collect info of which client is making request

    user_name = "test"
    server_url = "http://127.0.0.1:8091/mcp" # fetched from database against username

    transport = StreamableHttpTransport(url=server_url)
    mcp_client = Client(transport)

    # Parse model identifier
    try:
        provider, model_name = parse_model_identifier(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract API key from various sources
    api_key = request.api_key
    
    if not api_key and authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    if not api_key and x_api_key:
        api_key = x_api_key
    
    if not api_key:
        raise HTTPException(
            status_code=401, 
            detail="API key required. Provide it in request body, Authorization header, or X-API-Key header"
        )
    
    # Create client
    try:
        client = create_client(provider, api_key, request.base_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create client: {str(e)}")
    

    tools = await get_mcp_tools(provider, mcp_client)
    
    # Let llm decide tool call
    function_to_call = {}
    function_response = None

    # Limit message hsitory
    latest_user_message = request.messages[-1] if request.messages else None
    if latest_user_message:
        request.messages = [latest_user_message]
    else:
        # Fallback if no messages exist
        request.messages = []
    
    if tools:  # Only attempt function calling if tools are available         
        try:
            # Get only the latest user message instead of full history
            latest_message = request.messages[-1] if request.messages else None
            
            if not latest_message:
                logger.warning("No messages found in request")
                return
                
            if provider == "openai":
                # Use only the latest message for OpenAI
                formatted_messages = [{"role": latest_message.role, "content": latest_message.content}]
                                
                response = await client.client.chat.completions.create(
                    model=model_name,
                    messages=formatted_messages,
                    tools=[{"type": "function", "function": tool} for tool in tools],
                    tool_choice="auto"
                )
                                
                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    function_to_call["function_name"] = tool_call.function.name
                    function_to_call['function_args'] = json.loads(tool_call.function.arguments)
            
            elif provider == "anthropic":
                # For Anthropic, create a minimal message list with just the latest message
                minimal_messages = [latest_message]
                system_message, formatted_messages = client._format_messages(minimal_messages)
                
                response = await client.client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    tools=tools,
                    messages=formatted_messages,
                    system=system_message if system_message else None
                )
                                
                # Check if there's a tool use in the response
                for content in response.content:
                    if hasattr(content, 'type') and content.type == 'tool_use':
                        function_to_call["function_name"] = content.name
                        function_to_call["function_args"] = content.input
            
            elif provider == "gemini":
                tool_list = types.Tool(function_declarations=tools)
                config = types.GenerateContentConfig(tools=[tool_list])
                                
                # Format only the latest message for Gemini
                minimal_messages = [latest_message]
                prompt = client._format_messages(minimal_messages)
                                
                response = await asyncio.to_thread(
                    client.client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                
                if response.candidates[0].content.parts[0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    function_to_call["function_name"] = function_call.name
                    function_to_call["function_args"] = dict(function_call.args)
                            
        except Exception as e:
            logger.error(f"Function calling error for {provider}: {str(e)}")
            # Continue without function calling if there's an error


    # Execute function if one was called
    if function_to_call.get("function_name"):
        try:
            function_response = await execute_mcp_function(mcp_client, function_to_call)
            
            # Format messages with function call and response
            request.messages = format_messages_with_function_call(
                provider, 
                request.messages, 
                function_to_call, 
                function_response
            )
            
        except Exception as e:
            logger.error(f"Function execution error: {str(e)}")
            # Add error context to messages
            error_message = ChatMessage(
                role="user",
                content=f"There was an error executing the function {function_to_call.get('function_name', 'unknown')}: {str(e)}. Tell there was an error in accessing knowledge base, nothing else. Just state 'Sorry it cant be accessed at the moment. Please, try agian'."
            )
            request.messages.append(error_message)

    # Prepare kwargs
    kwargs = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    }

    if request.stream:
        # Streaming response
        async def generate():
            # Send initial chunk
            yield "data: " + json.dumps({
                "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }) + "\n\n"
            
            # Stream content
            try:
                async for chunk in client.chat_stream(model_name, request.messages, **kwargs):
                    if chunk.strip() and not chunk.startswith("Error:"):
                        yield "data: " + json.dumps({
                            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }) + "\n\n"
                    elif chunk.startswith("Error:"):
                        # Send error as final chunk
                        yield "data: " + json.dumps({
                            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": "stop"
                            }]
                        }) + "\n\n"
                        break
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield "data: " + json.dumps({
                    "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "stop"
                    }]
                }) + "\n\n"
            
            # End of stream
            yield "data: " + json.dumps({
                "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }) + "\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
    else:
        # Non-streaming response
        try:
            result = await client.chat_complete(model_name, request.messages, **kwargs)
            
            # Basic response structure
            response_choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": "stop"
            }
            
            return {
                "id": f"chatcmpl-{int(datetime.now().timestamp())}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "provider": provider,
                "choices": [response_choice],
                "usage": result.get("usage", {}),
                "function_call_executed": function_to_call.get("function_name") if function_to_call else None
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Completion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "providers": list(MODEL_REGISTRY.keys()),
        "total_models": sum(len(models) for models in MODEL_REGISTRY.values()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/v1/models/{provider}")
async def list_provider_models(provider: str):
    """List models for a specific provider"""
    if provider not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Provider {provider} not supported")
    
    models = []
    for model_id, model_info in MODEL_REGISTRY[provider].items():
        models.append({
            "id": f"{provider}/{model_id}",
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": provider,
            "provider": provider,
            "name": model_info["name"],
            "description": model_info["description"]
        })
    
    return {"object": "list", "data": models}

async def run():
    server_url = "http://127.0.0.1:8091/mcp" # fetched from database against username

    transport = StreamableHttpTransport(url=server_url)
    mcp_client = Client(transport)
    tools =  await get_mcp_tools("gemini", mcp_client)
    print(tools)
if __name__ == "__main__":
    print("Starting OpenWebUI Multi-Provider LLM Server")
    print(f"Supporting {sum(len(models) for models in MODEL_REGISTRY.values())} models across {len(MODEL_REGISTRY)} providers")
    print("Clients must provide their own API keys")
    
    
    asyncio.run(run())
    # uvicorn.run(
    #     "main:app",
    #     host=os.getenv("HOST", "0.0.0.0"),
    #     port=int(os.getenv("PORT", "8082")),
    #     reload=os.getenv("RELOAD", "false").lower() == "true",
    #     access_log=True
    # )