import numpy as np
import logging
import os
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from extensions.app.config import Config


logger = logging.getLogger(__name__)

class ModelManager:
    """Manages LLM and embedding models with auto-detection from config"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = None
    
    @classmethod
    def from_config_file(cls, config_path: str = None):
        """Create ModelManager by auto-detecting config file and binding"""
        if config_path is None:
            # Try to find config file in common locations
            possible_paths = [
                "./config.json",
                "./extensions/app/config.json",
                os.getenv("CONFIG_PATH", "")
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                raise FileNotFoundError("Config file not found. Please specify config_path or set CONFIG_PATH env var")
        
        # Load config directly from JSON if Config class is not available
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create a simple config object
            class SimpleConfig:
                def __init__(self, config_dict):
                    self.__dict__.update(config_dict)
            
            config = SimpleConfig(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
        
        return cls(config)
    
    async def get_llm_model_func(self):
        """Create LLM model function based on config - maintains exact same interface"""
        binding = self.config.llm.get("binding", "").lower()
        
        async def llm_model_func(
            prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
        ) -> str:
            if binding == "google_genai":
                return await self._google_genai_complete(
                    prompt, system_prompt, history_messages, **kwargs
                )
            elif binding == "openai":  # Pure OpenAI
                return await self._openai_complete(
                    prompt, system_prompt, history_messages, **kwargs
                )
            elif binding == "openai_compatible":  # OpenAI-compatible APIs like DeepSeek
                return await self._openai_compatible_complete(
                    prompt, system_prompt, history_messages, **kwargs
                )
            elif binding == "ollama":
                return await self._ollama_complete(
                    prompt, system_prompt, history_messages, **kwargs
                )
            else:
                # Fallback: try to auto-detect from environment or config
                return await self._auto_detect_and_complete(
                    prompt, system_prompt, history_messages, **kwargs
                )
        
        return llm_model_func

    def _filter_openai_kwargs(self, kwargs):
        """Filter out LightRAG-specific kwargs that OpenAI doesn't support"""
        # List of parameters that OpenAI API supports
        supported_params = {
            'max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty',
            'stop', 'stream', 'logit_bias', 'user', 'response_format', 'tools', 'tool_choice',
            'parallel_tool_calls', 'seed', 'logprobs', 'top_logprobs', 'n'
        }
        
        # Filter kwargs to only include supported parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        # Log filtered out parameters for debugging
        filtered_out = {k: v for k, v in kwargs.items() if k not in supported_params}
        if filtered_out:
            logger.debug(f"Filtered out unsupported OpenAI parameters: {list(filtered_out.keys())}")
        
        return filtered_kwargs

    async def _google_genai_complete(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Handle Google GenAI completion"""
        client = genai.Client(api_key=self.config.llm["api_key"])
        
        if history_messages is None:
            history_messages = []

        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"

        for msg in history_messages:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        combined_prompt += f"user: {prompt}"

        response = client.models.generate_content(
            model=self.config.llm["model"],
            contents=[combined_prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.llm.get("max_tokens", 1024),
                temperature=self.config.llm.get("temperature", 0.2)
            ),
        )
        return response.text

    async def _openai_complete(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Handle pure OpenAI completion"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI bindings")
        
        client = AsyncOpenAI(
            api_key=self.config.llm.get("api_key") or os.getenv("OPENAI_API_KEY")
        )
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history_messages:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": prompt})
        
        # Filter out LightRAG-specific kwargs
        filtered_kwargs = self._filter_openai_kwargs(kwargs)
        
        response = await client.chat.completions.create(
            model=self.config.llm.get("model", "gpt-4o-mini"),
            messages=messages,
            max_tokens=self.config.llm.get("max_tokens", 1024),
            temperature=self.config.llm.get("temperature", 0.2),
            **filtered_kwargs
        )
        
        return response.choices[0].message.content

    async def _openai_compatible_complete(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Handle OpenAI-compatible API completion (DeepSeek, etc.)"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI-compatible bindings")
        
        client = AsyncOpenAI(
            api_key=self.config.llm.get("api_key") or os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.llm.get("base_url", os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"))
        )
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history_messages:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": prompt})
        
        # Filter out LightRAG-specific kwargs
        filtered_kwargs = self._filter_openai_kwargs(kwargs)
        
        response = await client.chat.completions.create(
            model=self.config.llm.get("model", os.getenv("LLM_MODEL", "deepseek-chat")),
            messages=messages,
            max_tokens=self.config.llm.get("max_tokens", 1024),
            temperature=self.config.llm.get("temperature", 0.2),
            **filtered_kwargs
        )
        
        return response.choices[0].message.content

    async def _ollama_complete(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Handle Ollama completion"""
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is required for Ollama bindings")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history_messages:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": prompt})
        
        client = ollama.AsyncClient(
            host=self.config.llm.get("host", os.getenv("LLM_BINDING_HOST", "http://localhost:11434"))
        )
        
        response = await client.chat(
            model=self.config.llm.get("model", os.getenv("LLM_MODEL", "qwen2.5-coder:7b")),
            messages=messages,
            options={
                "num_ctx": self.config.llm.get("max_token_size", 8192),
                "temperature": self.config.llm.get("temperature", 0.2),
                **self.config.llm.get("options", {}),
                **(self.config.llm.get("llm_model_kwargs", {}).get("options", {}))
            }
        )
        
        return response['message']['content']

    async def _auto_detect_and_complete(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        """Auto-detect provider from environment variables"""
        # Check environment variables to determine provider
        if os.getenv("GOOGLE_API_KEY") or hasattr(self.config, 'llm') and 'google' in str(self.config.llm.get('api_key', '')).lower():
            return await self._google_genai_complete(prompt, system_prompt, history_messages, **kwargs)
        elif os.getenv("LLM_BINDING_HOST", "").startswith("http://localhost:11434") or os.getenv("LLM_MODEL", "").endswith(":latest"):
            return await self._ollama_complete(prompt, system_prompt, history_messages, **kwargs)
        elif os.getenv("LLM_BINDING_HOST") and "deepseek" in os.getenv("LLM_BINDING_HOST", ""):
            return await self._openai_compatible_complete(prompt, system_prompt, history_messages, **kwargs)
        else:
            # Default to OpenAI
            return await self._openai_complete(prompt, system_prompt, history_messages, **kwargs)

    async def get_embedding_func(self):
        """Create embedding function based on config - maintains exact same interface"""
        binding = self.config.embedding.get("binding", "").lower()
        
        if binding == "sentence_transformers":
            return await self._get_sentence_transformers_func()
        elif binding == "openai":
            return await self._get_openai_embedding_func()
        elif binding == "openai_compatible":
            return await self._get_openai_compatible_embedding_func()
        elif binding == "ollama":
            return await self._get_ollama_embedding_func()
        else:
            # Auto-detect from environment
            return await self._auto_detect_embedding_func()

    def _filter_openai_embedding_kwargs(self, kwargs):
        """Filter out kwargs that OpenAI embedding API doesn't support"""
        # Parameters that OpenAI embedding API supports
        supported_params = {
            'input', 'model', 'encoding_format', 'dimensions', 'user'
        }
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        # Log filtered out parameters for debugging
        filtered_out = {k: v for k, v in kwargs.items() if k not in supported_params}
        if filtered_out:
            logger.debug(f"Filtered out unsupported OpenAI embedding parameters: {list(filtered_out.keys())}")
        
        return filtered_kwargs

    async def _get_sentence_transformers_func(self):
        """Get sentence transformers embedding function"""
        if self.embedding_model is None:
            model_name = self.config.embedding.get("model_name", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            
            # Log actual embedding dimension
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_embedding.shape[1]
            expected_dim = self.config.embedding.get("embedding_dim", 384)
            
            logger.info(f"Model: {model_name}, Actual dim: {actual_dim}, Expected dim: {expected_dim}")
            
            if actual_dim != expected_dim:
                logger.warning(f"Embedding dimension mismatch! Model produces {actual_dim}D embeddings but config expects {expected_dim}D")

        async def embedding_func(texts: list[str]) -> np.ndarray:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings

        return embedding_func

    async def _get_openai_embedding_func(self):
        """Get OpenAI embedding function"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")
        
        client = AsyncOpenAI(
            api_key=self.config.embedding.get("api_key") or os.getenv("OPENAI_API_KEY")
        )

        async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
            # Filter out unsupported parameters
            filtered_kwargs = self._filter_openai_embedding_kwargs(kwargs)
            
            response = await client.embeddings.create(
                model=self.config.embedding.get("model_name", "text-embedding-3-small"),
                input=texts,
                **filtered_kwargs
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings

        return embedding_func

    async def _get_openai_compatible_embedding_func(self):
        """Get OpenAI-compatible embedding function"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI-compatible embeddings")
        
        client = AsyncOpenAI(
            api_key=self.config.embedding.get("api_key") or os.getenv("LLM_BINDING_API_KEY"),
            base_url=self.config.embedding.get("base_url", os.getenv("EMBEDDING_BINDING_HOST"))
        )

        async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
            # Filter out unsupported parameters
            filtered_kwargs = self._filter_openai_embedding_kwargs(kwargs)
            
            response = await client.embeddings.create(
                model=self.config.embedding.get("model_name", "text-embedding-3-small"),
                input=texts,
                **filtered_kwargs
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings

        return embedding_func

    async def _get_ollama_embedding_func(self):
        """Get Ollama embedding function"""
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is required for Ollama embeddings")
        
        host = self.config.embedding.get("host", os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"))
        client = ollama.AsyncClient(host=host)

        async def embedding_func(texts: list[str], **kwargs) -> np.ndarray:
            embeddings = []
            model_name = self.config.embedding.get("model_name", os.getenv("EMBEDDING_MODEL", "bge-m3:latest"))
            for text in texts:
                response = await client.embeddings(
                    model=model_name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            return np.array(embeddings)

        return embedding_func

    async def _auto_detect_embedding_func(self):
        """Auto-detect embedding provider"""
        # Check for Ollama first
        if os.getenv("EMBEDDING_BINDING_HOST", "").startswith("http://localhost:11434") or \
           os.getenv("EMBEDDING_MODEL", "").endswith(":latest"):
            return await self._get_ollama_embedding_func()
        # Check for OpenAI
        elif os.getenv("OPENAI_API_KEY"):
            return await self._get_openai_embedding_func()
        else:
            # Default to sentence transformers
            return await self._get_sentence_transformers_func()

    def get_lightrag_params(self):
        """Get LightRAG initialization parameters"""
        params = {}
        
        # Add Ollama-specific parameters if needed
        if self.config.llm.get("binding") == "ollama":
            params.update({
                "llm_model_name": self.config.llm.get("model", os.getenv("LLM_MODEL", "qwen2.5-coder:7b")),
                "llm_model_max_token_size": self.config.llm.get("max_token_size", 8192),
                "llm_model_kwargs": {
                    "host": self.config.llm.get("host", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")),
                    "options": {"num_ctx": self.config.llm.get("max_token_size", 8192)},
                    "timeout": self.config.llm.get("timeout", int(os.getenv("TIMEOUT", "300"))),
                    **self.config.llm.get("llm_model_kwargs", {})
                }
            })
        
        return params


# Convenience functions that maintain your existing interface
async def get_model_functions(config_path: str = None):
    """Get model functions without changing your existing code"""
    manager = ModelManager.from_config_file(config_path)
    
    llm_func = await manager.get_llm_model_func()
    embedding_func = await manager.get_embedding_func()
    lightrag_params = manager.get_lightrag_params()
    
    return llm_func, embedding_func, lightrag_params