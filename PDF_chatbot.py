import io
from typing import List
from PyPDF2 import PdfReader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

Example of your response should be:

The answer is foo
SOURCES: xyz


Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}
