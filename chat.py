from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from decouple import Config, RepositoryEnv
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Dict
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from models import User

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from models import ChatSchema
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse,HTMLResponse
import logging
from fastapi import APIRouter


cache_instance=InMemoryCache()
set_llm_cache(cache_instance)
config = Config(RepositoryEnv("C:/Users/DELL/Desktop/llm-experiments/.env"))

COHERE_API_KEY = config('COHERE_API_KEY')
MONGO_URI = config('MONGO_URI')
os.environ["COHERE_API_KEY"] = config('COHERE_API_KEY')
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = config('LANGCHAIN_API_KEY')  # Update with your API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="new-bino"

router = APIRouter()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)
# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create a vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever
retriever = vectorstore.as_retriever()

chat = ChatCohere(model="command", temperature=0.5)
SYSTEM_TEMPLATE = """
You are a friendly, helpful, respectful, and honest conversational chatbot named Bino

Bino should Decide if the user's question requires a documentation lookup, only if yes should you Answer the question based on the provided documents, if not answer the question without looking at the document. Your responses should adhere to the following guidelines:
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not try to learn about any topic by asking the user questions.
- do not take advice from the user.
- do not acknowledge learning anything from the user.
- dont acknowledge any information learnt from the user


If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""


# SYSTEM_TEMPLATE = """
# Answer the user's questions based on the below context. 
# If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

# <context>
# {context}
# </context>
# """
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)



def parse_retriever_input(params: Dict):
    return params["history"][-1].content


retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else."

        ),
    ]
)

query_transformation_chain = query_transform_prompt | chat

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("history", [])) == 1,
        # If only one message, then we just pass that message's content to retriever
        (lambda x: x["history"][-1].content) | retriever,
    ),
    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    query_transform_prompt | chat| StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")


conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)




chain=conversational_retrieval_chain | question_answering_prompt |  chat



@router.post("/chat/{session_id}")
async def chat(session_id:str, message:ChatSchema):
    try:
        with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="MONGO_URI",
        database_name="fastapi_mongo",
        collection_name="chat_histories",
        ),
        input_messages_key="question",
        history_messages_key="history",
        )
  
        
        response = with_message_history.invoke(
                {"question": f"{message.query}"},
                config={"configurable": {"session_id": session_id}}
                )
        return JSONResponse(content={"bot":f" {response}"},
                                    status_code= 200
                    )
    except Exception as e:
            logging.error(f"Error in assistant communication: {e}")
            raise HTTPException(
                    status_code=400,detail=f"an error occoured while starting the assistant {e}"
                )


