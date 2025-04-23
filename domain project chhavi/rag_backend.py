from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # loads OPENAI_API_KEY from .env

def hr_index():
    data_load = PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10
    )

    data_embeddings = OpenAIEmbeddings()  # Uses `text-embedding-ada-002`

    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    db_index = data_index.from_loaders([data_load])
    return db_index


def hr_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=300
    )


def hr_rag_response(index, question):
    rag_llm = hr_llm()
    result = index.query(question=question, llm=rag_llm)
    return result
