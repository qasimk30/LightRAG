from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime

# Configuration Models
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class OpenWebUIRequest(BaseModel):
    model: str = Field(..., description="Model identifier with provider prefix (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for response generation")
    stream: bool = Field(default=True, description="Whether to stream the response")
    
    # OpenWebUI specific fields
    api_key: Optional[str] = Field(default=None, description="API key for the model provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for the provider")
    
    @validator('model')
    def validate_model(cls, v):
        if '/' not in v:
            raise ValueError("Model must include provider prefix (e.g., 'openai/gpt-4')")
        return v

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str
    provider: str
    description: Optional[str] = None
