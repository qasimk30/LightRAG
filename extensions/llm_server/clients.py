from openai import AsyncOpenAI
from typing import AsyncGenerator, Dict, Any, Optional, List
import logging
from fastapi import HTTPException
import anthropic
from openai import AsyncOpenAI
import google.generativeai as gai
from google import genai
import asyncio

from extensions.llm_server.models import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Registry - Static list of supported models
MODEL_REGISTRY = {
    "openai": {
        "gpt-4": {"name": "GPT-4", "description": "Most capable GPT-4 model"},
        "gpt-4-turbo": {"name": "GPT-4 Turbo", "description": "Latest GPT-4 Turbo model"},
        "gpt-4o": {"name": "GPT-4o", "description": "GPT-4 Omni model"},
        "gpt-4o-mini": {"name": "GPT-4o Mini", "description": "Smaller, faster GPT-4o model"},
        "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "description": "Fast and efficient model"},
        "o1-preview": {"name": "o1 Preview", "description": "Advanced reasoning model"},
        "o1-mini": {"name": "o1 Mini", "description": "Smaller reasoning model"},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"name": "Claude 3.5 Sonnet", "description": "Most intelligent Claude model"},
        "claude-3-5-haiku-20241022": {"name": "Claude 3.5 Haiku", "description": "Fast and lightweight Claude model"},
        "claude-3-opus-20240229": {"name": "Claude 3 Opus", "description": "Most powerful Claude model"},
        "claude-3-sonnet-20240229": {"name": "Claude 3 Sonnet", "description": "Balanced Claude model"},
        "claude-3-haiku-20240307": {"name": "Claude 3 Haiku", "description": "Fastest Claude model"},
    },
    "gemini": {
        "gemini-1.5-pro": {"name": "Gemini 1.5 Pro", "description": "Most capable Gemini model"},
        "gemini-1.5-flash": {"name": "Gemini 1.5 Flash", "description": "Fast Gemini model"},
        "gemini-1.5-flash-8b": {"name": "Gemini 1.5 Flash 8B", "description": "Lightweight Gemini model"},
        "gemini-pro": {"name": "Gemini Pro", "description": "Standard Gemini model"},
    }
}

# LLM Clients
class OpenAIClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1"
        )
    
    async def chat_stream(self, model: str, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        try:
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=True,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def chat_complete(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        try:
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"OpenAI completion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

class AnthropicClient:
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    def _format_messages(self, messages: List[ChatMessage]) -> tuple:
        """Format messages for Anthropic API"""
        system_message = ""
        formatted_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                formatted_messages.append({"role": msg.role, "content": msg.content})
        
        return system_message, formatted_messages
    
    async def chat_stream(self, model: str, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        try:
            system_message, formatted_messages = self._format_messages(messages)
            
            async with self.client.messages.stream(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                system=system_message if system_message else None,
                messages=formatted_messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def chat_complete(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        try:
            system_message, formatted_messages = self._format_messages(messages)
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                system=system_message if system_message else None,
                messages=formatted_messages
            )
            
            return {
                "content": response.content[0].text,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Anthropic completion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Anthropic API error: {str(e)}")

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Configure for this instance
        gai.configure(api_key=api_key)
        self.client = genai.Client(api_key=api_key)
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for Gemini API"""
        formatted_parts = []
        print(messages)
        for msg in messages:
            role_prefix = {
                "system": "System: ",
                "user": "User: ",
                "assistant": "Assistant: "
            }.get(msg.role, f"{msg.role.title()}: ")
            formatted_parts.append(f"{role_prefix}{msg.content}")
        return "\n\n".join(formatted_parts)
    
    async def chat_stream(self, model: str, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        try:
            # Create a fresh model instance for each request
            model_instance = gai.GenerativeModel(model)
            prompt = self._format_messages(messages)
            
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                stream=True,
                generation_config=gai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7)
                )
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def chat_complete(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        try:
            model_instance = gai.GenerativeModel(model)
            prompt = self._format_messages(messages)
            
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                generation_config=gai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7)
                )
            )
            
            return {
                "content": response.text,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0
                }
            }
        except Exception as e:
            logger.error(f"Gemini completion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
