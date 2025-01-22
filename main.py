import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

load_dotenv()

# Load and process PDF documents
def load_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Language Model
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-large", 
#     model_kwargs={"temperature": 0.5, "max_length": 512}
# )
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large", 
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_TOKEN"),
    model_kwargs={"temperature":0.5, "max_length":512}
)

# Create a more robust retrieval QA chain
def setup_qa_chain(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

# Process documents and create vector store
documents = load_pdfs('papers/')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(texts, embeddings)

# Setup QA Chain
qa_chain = setup_qa_chain(vectorstore)

# Query function
def query_papers(question):
    result = qa_chain({"query": question})
    return result['result']

# Summarization function
def summarize_paper(paper_title):
    query = f"Summarize the key points of the paper titled '{paper_title}'"
    return query_papers(query)

# Example Usage
question = "What are the skills of this person?"
answer = query_papers(question)
print(f"Question: {question}\nAnswer: {answer}")
