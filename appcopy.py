import streamlit as st
from dotenv import load_dotenv
import openai
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import re
import spacy
import subprocess
from datetime import datetime

# Load environment variables
load_dotenv()

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download SpaCy model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Set page configuration
st.set_page_config(page_title="ResuBot", page_icon="🤖", layout="centered")

# Custom CSS to style the app
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 20px;
        }
        .chat-bubble-user {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 70%;
            word-wrap: break-word;
            text-align: left;
            align-self: flex-start;
        }
        .chat-bubble-bot {
            background-color: #f1f0f0;
            color: #333333;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 10px 0;
            max-width: 70%;
            word-wrap: break-word;
            text-align: left;
            align-self: flex-end;
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .stTextInput > div > input {
            border-radius: 15px;
            border: 2px solid #007bff;
            padding: 15px;
            width: 100%;
        }
        .stApp {
            align-items: center;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Add a sidebar with more information
with st.sidebar:
    st.header("About ResuBot 🤖")
    st.write("""
        *ResuBot* is your personal assistant for interacting with resumes.
        Upload a resume in PDF format, and ask questions to get insights or 
        information as if you're conversing with the actual candidate.
        
        Created with:
        - *Streamlit*
        - *LangChain*
        - *OpenAI*
    """)
    st.write("Developed by *Sayinah Ali*")
    st.markdown("<hr>", unsafe_allow_html=True)

# Function to extract name using OpenAI
def extract_name_with_openai(text):
    # Prepare the prompt for the new API
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts specific information from text."
        },
        {
            "role": "user",
            "content": (
                "Extract the candidate's full name from the following resume text:\n\n"
                f"{text}\n\n"
                "Return only the name in the format 'First Last'."
            )
        }
    ]
    
    # Use the new openai.Chat API to get the response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=20,
        temperature=0.0
    )
    
    # Extract the name from the response
    name = response.choices[0].message['content'].strip()
    
    return name

# Function to check if the document is a resume
def check_if_resume(text):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that validates if a text is likely to be a resume."
        },
        {
            "role": "user",
            "content": (
                "Based on the following text, determine if this is likely a resume:\n\n"
                f"{text}\n\n"
                "Respond with 'Yes' or 'No'."
            )
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=10,
        temperature=0.0
    )
    
    validation = response.choices[0].message['content'].strip().lower()
    return validation == "yes"

# Function to extract and calculate experience and project details in the exact format
def calculate_experience_and_projects(text):
    # Prepare the prompt for extracting experience and project information
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts specific sections of resumes."
        },
        {
            "role": "user",
            "content": (
                "From the following resume text, identify and list the project names, "
                "calculate the total number of projects mentioned, and determine the total work experience duration "
                "in the format:\n\n"
                "The resume mentions X projects:\n"
                "- Project 1\n"
                "- Project 2\n"
                "- Project 3\n\n"
                "The total work experience is from [start date] to [end date], which is approximately Y years or Z months."
                f"\n\n{text}"
            )
        }
    ]
    
    # Use OpenAI to extract and calculate the required information
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.0
    )
    
    # Extract and return the result
    result = response.choices[0].message['content'].strip()
    return result

    
def main():
    st.markdown("<div class='main-title'>ResuBot 🤖</div>", unsafe_allow_html=True)
    st.markdown("""
    *ResuBot*: Your personal assistant for interacting with resumes.
    
    Upload a resume in PDF format and converse with the bot to get insights or information as if talking to the actual candidate.
    """)
    
    add_vertical_space(3)

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Upload a PDF file
    pdf = st.file_uploader("Upload your resume PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if not check_if_resume(text):
            st.error("This is not a resume. Please upload the correct file.")
            return  # Stop the execution here if not a resume

        # Use OpenAI to extract the candidate's name
        name = extract_name_with_openai(text)
        if name:
            st.success(f"This is the resume of {name}.")
        else:
            st.warning("Name not found in the resume. Please make sure the resume has a clear name section.")

        # Calculate experience and project details using OpenAI in the exact format
        experience_and_projects = calculate_experience_and_projects(text)
        
        # Display the extracted data
        st.subheader("Resume Summary")
        st.write(experience_and_projects)

        # Form to accept user questions/query
        with st.form(key='query_form'):
            query = st.text_input("Ask a question about the resume:", key="query_input")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and query:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Embeddings
            store_name = pdf.name[:-4]
            st.write(f'Processing {store_name}...')

            # When creating the index:
            if os.path.exists(f"{store_name}.index"):
                VectorStore = FAISS.load_local(f"{store_name}", OpenAIEmbeddings())
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                VectorStore.save_local(f"{store_name}")

            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Update conversation history
            st.session_state.conversation.append({"role": "user", "content": query})
            st.session_state.conversation.append({"role": "bot", "content": response})

        # Display conversation history
        for chat in st.session_state.conversation:
            if chat["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'><strong>User:</strong><br>{chat['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-bot'><strong>ResuBot:</strong><br>{chat['content']}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()