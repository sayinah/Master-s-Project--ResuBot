import streamlit as st
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

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
        /* Style the main title */
        .main-title {
            font-size: 2.5em;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 20px;
        }
        
        /* Style for the chat bubbles */
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

        /* Add some padding to the sidebar */
        .sidebar .sidebar-content {
            padding: 20px;
        }

        /* Style for the clear chat button */
        .stButton button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: 0.3s;
        }

        .stButton button:hover {
            background-color: #ff3333;
        }

        /* Style the input box */
        .stTextInput > div > input {
            border-radius: 15px;
            border: 2px solid #007bff;
            padding: 15px;
            width: 100%;
        }

        /* Centered layout */
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
        **ResuBot** is your personal assistant for interacting with resumes.
        Upload a resume in PDF format, and ask questions to get insights or 
        information as if you're conversing with the actual candidate.
        
        Created with:
        - **Streamlit**
        - **LangChain**
        - **OpenAI**
    """)
    st.write("Developed by **Sayinah Ali**")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Use the 'Clear Chat' button to reset the conversation at any time.")

# Main app function
def extract_name(text):
    # Using SpaCy to extract names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def check_if_resume(text):
    # Improved check for resume keywords
    keywords = ["experience", "education", "skills", "summary", "objective", "work history", "certifications"]
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
    return keyword_count >= 2  # Adjust threshold as needed

def extract_resume_sections(text):
    # Extract sections like experience, education, skills
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "summary": "",
        "objective": "",
        "certifications": ""
    }
    for section in sections.keys():
        pattern = re.compile(rf"{section}[:\s]", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            sections[section] = text[match.start():].split('\n\n', 1)[0].strip()
    return sections

def extract_experience_details(text):
    experience_sections = []
    experience_pattern = re.compile(r"(experience|work history|employment history)[:\s]", re.IGNORECASE)
    matches = list(experience_pattern.finditer(text))
    if matches:
        for match in matches:
            start_idx = match.end()
            end_idx = text.find('\n\n', start_idx)
            experience_section = text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
            experience_sections.append(experience_section)
    return experience_sections

def extract_projects(text):
    projects = []
    project_pattern = re.compile(r"(projects|responsibilities|achievements)[:\s]", re.IGNORECASE)
    matches = list(project_pattern.finditer(text))
    if matches:
        for match in matches:
            start_idx = match.end()
            end_idx = text.find('\n\n', start_idx)
            project_section = text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
            projects.append(project_section)
    return projects

def parse_experience(experience_section):
    experience_details = []
    experience_lines = experience_section.split('\n')
    for line in experience_lines:
        dates = re.findall(r"\b\d{4}\b", line)
        if dates:
            experience_details.append({
                'line': line,
                'years': [int(date) for date in dates]
            })
    return experience_details

def main():
    st.markdown("<div class='main-title'>ResuBot 🤖</div>", unsafe_allow_html=True)
    st.markdown("""
    **ResuBot**: Your personal assistant for interacting with resumes.
    
    Upload a resume in PDF format and converse with the bot to get insights or information as if talking to the actual candidate.
    """)
    
    add_vertical_space(3)

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Add a button to clear the conversation
    if st.button("Clear Chat"):
        st.session_state.conversation = []  # Clear the conversation history

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

        name = extract_name(text)
        if name:
            st.success(f"This is the resume of {name}.")
        else:
            st.warning("Name not found in the resume. Please make sure the resume has a clear name section.")

        experience_sections = extract_experience_details(text)
        project_sections = extract_projects(text)
        experience_details = [parse_experience(section) for section in experience_sections]

        # Proceed with embedding and question-answering setup
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

        # Form to accept user questions/query
        with st.form(key='query_form'):
            query = st.text_input("Ask a question about the resume:", key="query_input")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Improve QA logic for specific questions about experience and projects
            if "years of experience" in query.lower() or "projects" in query.lower():
                total_years = 0
                project_list = []

                for detail in experience_details:
                    for item in detail:
                        if len(item['years']) == 2:
                            total_years += item['years'][1] - item['years'][0]
                        project_list.extend([item['line']])

                response = f"The candidate has approximately {total_years} years of experience. They have worked on the following projects:\n" + "\n".join(project_list)

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


