# For query transformation
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

# For basic RAG implementation
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load documents
loader = PyPDFLoader("./italy_travel.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
documents = text_splitter.split_documents(documents)

# 2. Convert to vectors
embedder = OllamaEmbeddings(model="mistral:7b-instruct-q4_K_M")
#pages = []
#for doc in documents:
#    page = doc.page_content
#    pages.append(page)
#    print(page)
#print(len(pages))

#_ = embedder.embed_documents([doc.page_content for doc in documents])

# 3. Store in vector databaseo
print(3)
#vector_db = Chroma.from_documents(documents=documents,embedding=embedder,persist_directory=".")
vector_db = Chroma(persist_directory="db", embedding_function=embedder)

# 4. Retrieve similar docs
query = "Tell me about the Vatican City".upper()
results = vector_db.similarity_search(query)
print(results[0].page_content)