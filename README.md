# AI Document Assistant

AI Document Assistant is a Streamlit-based web application that allows users to upload PDF documents and interactively ask questions about their content using advanced AI-powered natural language processing. The app leverages LangChain, HuggingFace embeddings, FAISS vector store, and Groq's ChatGroq model to provide precise and context-aware answers.

## Features

- Upload and analyze multiple PDF documents simultaneously
- AI-powered question answering based on document content
- Source document tracking with page-level context
- Dark and light mode support with a modern UI
- Built with Streamlit, LangChain, and Groq AI models

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd AIMBot
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Run the Streamlit app:

```bash
streamlit run drbot.py
```

Open the URL provided by Streamlit (usually http://localhost:8501) in your browser.

Upload one or more PDF documents using the sidebar, then ask questions about the documents in the chat interface.

## How It Works

- Uploaded PDFs are read and their text extracted page by page.
- Text is split into chunks and embedded using HuggingFace's sentence-transformers.
- A FAISS vector store is created for efficient similarity search.
- User questions are answered by retrieving relevant document chunks and passing them to the Groq Chat model with a custom prompt.
- Answers are displayed along with source document excerpts for transparency.

## Developer

- Shabarish B L
- GitHub: [Shabarish5](https://github.com/shabarish5)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
