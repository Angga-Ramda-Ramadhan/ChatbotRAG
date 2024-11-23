
from langchain.schema import Document  # Import Document class
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

load_dotenv()

os.environ.get("LANGCHAIN_TRACING_V2")
os.environ.get("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama3-8b-8192")

# Load your ChromaDB or retriever
# Load vectorstore with dynamic file path
def load_vectorstore(file_path):
    docs = extract_text_from_pdf(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    text_chunks = text_splitter.split_text(docs)

    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    return vectorstore.as_retriever()


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    extracted_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return extracted_text

def process_prompt(file_path, prompt_user):
    retriever = load_vectorstore(file_path)
    system_prompt = (
        "You need to read all the text in the pdf file and give the specific answer from user prompting. Make sure you serve a relevant answer. Also give a related answer with indonesian language."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": prompt_user})
    return response["answer"]


chat_interface = gr.Interface(
    fn=process_prompt,
    inputs=[
        gr.File(label="Upload PDF"),  # Input file
        gr.Textbox(label="Your Question"),  # Input prompt
    ],
    outputs="text"
)

chat_interface.launch(server_name='0.0.0.0', server_port=8080)

