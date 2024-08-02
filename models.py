from pydantic import BaseModel, Field
from datetime import datetime, timedelta

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