# ResuBot 🤖

ResuBot is a personal assistant designed to interact with resumes. Upload a resume in PDF format, and ask questions to get insights or information as if you were conversing with the actual candidate.

## Features
- **PDF Upload:** Upload a resume in PDF format.
- **Name Extraction:** Automatically extracts the candidate's name from the resume.
- **Resume Verification:** Checks if the uploaded document is indeed a resume.
- **Experience Details:** Extracts and interprets experience details, including total years of experience.
- **Project Extraction:** Extracts project-related information from the resume.
- **Interactive Q&A:** Ask questions about the resume, and the bot will answer based on the content.

## Tech Stack
- **Streamlit:** For building the web interface.
- **LangChain:** For handling natural language processing and interactions.
- **OpenAI API:** For generating responses.
- **FAISS:** For efficient similarity search.
- **SpaCy:** For natural language processing tasks like name entity recognition.
- **PyPDF2:** For extracting text from PDF files.

## Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- Pip (Python package installer)

### Step 1: Clone the Repository
Clone the repository to your local machine using the following command:
git clone https://github.com/username/repository.git
cd repository

### Step 2: Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Step 3: Install Dependencies
Install the necessary Python packages using the requirements.txt file.
pip install -r requirements.txt

### Step 4: Set Up Environment Variables
Create a .env file in the root directory and add your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here

### Step 5: Run the Application
Start the Streamlit application using the following command:
streamlit run app.py
The app should now be running on http://localhost:8501.

### Usage
Upload a PDF: Upload a resume in PDF format via the app interface.
Ask Questions: Use the input box to ask questions about the resume. The bot will respond with relevant information.
Clear Chat: Use the "Clear Chat" button to reset the conversation at any time.

