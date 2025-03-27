from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv(override=True)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/", 
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)
embeddings = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    persist_directory=".chroma_db",
    collection_name="rag-chroma",
)

retriever = Chroma(
    embedding_function=embeddings,
    persist_directory=".chroma_db",
    collection_name="rag-chroma",
).as_retriever()  # so we can do similarity search


if __name__ == "__main__":
    # Persist the vector store to disk
    
    print(f"Vector store persisted.{vector_store = }")
