from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional
class User(BaseModel):
    id: str = Field(default=None)
    username: str = Field(...)
    password: str = Field(...)
    created_at: datetime = Field(default=datetime.utcnow())
    updated_at: datetime = Field(default=datetime.utcnow())

    
class ChatSchema(BaseModel):
    query:str

    class Config:
        json_schema_extra ={
            "example":{
                "query":"hi"
            
        }
        }

class ChatHistory(BaseModel):
    username: str
    conversation_id: str
    messages: List[dict] = Field(default_factory=list)