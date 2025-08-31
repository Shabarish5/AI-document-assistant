import os
import re
import streamlit as st
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    # Detect system theme preference
    import streamlit as st
    import platform

    # Default to light mode colors
    bg_color = "#090000"
    text_color = "#000000"
    accent_color = "#4A90E2"
    card_bg = "#000408"

    # Use CSS media query to detect system theme preference
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        * {{
            font-family: 'Inter', sans-serif;
        }}

        @media (prefers-color-scheme: dark) {{
            body {{
                background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%) !important;
                color: #ffffff !important;
            }}

            .stApp {{
                background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%) !important;
            }}

            .css-1d391kg, .css-12ttj6m {{
                background: #1e1e1e !important;
                color: #ffffff !important;
            }}

            .title {{
                background: linear-gradient(45deg, #00d4ff, #ff6b6b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            }}

            .chat-message {{
                background: #2d2d2d;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }}

            .sidebar {{
                background: #2d2d2d;
            }}

            .file-info {{
                background: #1e1e1e;
                border-left: 4px solid #00d4ff;
            }}

            .stButton>button {{
                background: linear-gradient(45deg, #00d4ff, #ff6b6b);
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }}

            .stButton>button:hover {{
                box-shadow: 0 8px 25px rgba(0,0,0,0.7);
            }}

            .stChatInput {{
                border: 2px solid #00d4ff;
            }}

            .loading {{
                border-top-color: #00d4ff;
            }}

            .expander {{
                background: #2d2d2d;
                border: 1px solid #00d4ff;
            }}
        }}

        @media (prefers-color-scheme: light) {{
            body {{
                background: linear-gradient(135deg, {bg_color} 0%, {card_bg} 100%) !important;
                color: {text_color} !important;
            }}

            .stApp {{
                background: linear-gradient(135deg, {bg_color} 0%, {card_bg} 100%) !important;
            }}

            .css-1d391kg, .css-12ttj6m {{
                background: {bg_color} !important;
                color: {text_color} !important;
            }}

            .title {{
                background: linear-gradient(45deg, {accent_color}, #ff6b6b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
            }}

            .chat-message {{
                background: {card_bg};
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}

            .sidebar {{
                background: {card_bg};
            }}

            .file-info {{
                background: {bg_color};
                border-left: 4px solid {accent_color};
            }}

            .stButton>button {{
                background: linear-gradient(45deg, {accent_color}, #ff6b6b);
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}

            .stButton>button:hover {{
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}

            .stChatInput {{
                border: 2px solid {accent_color};
            }}

            .loading {{
                border-top-color: {accent_color};
            }}

            .expander {{
                background: {card_bg};
                border: 1px solid {accent_color};
            }}
        }}

        .title {{
            font-size: 3.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 15px;
            margin-bottom: 2rem;
            animation: fadeInUp 1s ease-out;
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .chat-message {{
            font-size: 1.1rem;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 15px;
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }}

        .chat-message:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateX(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        .svg-icon {{
            width: 50px;
            height: 50px;
            filter: drop-shadow(0 0 10px {accent_color});
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}

        .sidebar {{
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            background: inherit !important;
        }}

        .upload-section {{
            margin-bottom: 20px;
        }}

        .stButton>button {{
            border-radius: 25px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}

        .stChatInput {{
            border-radius: 25px;
            padding: 10px 20px;
        }}

        .loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            animation: spin 1s ease-in-out infinite;
        }}

        /* Remove rectangle box in sidebar */

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="title">
            🤖 AI Document Assistant
            <!--<svg class="svg-icon" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                <path d="M0 0h24v24H0z" fill="none"/>
                <circle cx="12" cy="12" r="3" fill="{accent_color}"/>
            </svg>-->
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar content
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)

        with st.expander("ℹ️ About"):
            st.markdown("""
            **AI Document Assistant** helps you analyze PDF documents using advanced AI.

            **Features:**
            - 📚 Multi-document analysis
            - 🤖 AI-powered responses
            - 📊 Source document tracking

            Built with Streamlit & LangChain

            ---
            **Developer:**
            - Shabarish B L
            - GitHub: [Shabarish5](https://github.com/shabarish5)
            """)

        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents to get started",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_texts = []
                for uploaded_file in uploaded_files:
                    pdf_reader = PdfReader(uploaded_file)
                    for page_num, page in enumerate(pdf_reader.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            all_texts.append(f"Page {page_num}: {text}")

                combined_text = "\n".join(all_texts)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_text(combined_text)

                embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                st.session_state.vectorstore = FAISS.from_texts(docs, embedding_model)

                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")

                # Show file info
                for i, file in enumerate(uploaded_files):
                    st.markdown(f"""
                    <div class="file-info">
                        📄 <strong>{file.name}</strong><br>
                        Size: {file.size / 1024:.1f} KB
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        if st.session_state.vectorstore is None:
            st.error("⚠️ Please upload PDF documents first to start chatting!")
            st.info("💡 Upload documents in the sidebar to enable AI responses.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
            **Situation**
            You are a precise and knowledgeable information extraction assistant tasked with answering questions using only the provided context. The goal is to deliver clear, accurate information while ensuring technical concepts are explained in an accessible manner.

            **Task**
            - Carefully analyze the given context
            - Answer the specific question using ONLY information from the provided context
            - Present the answer in bulleted points
            - Translate technical terminology into simple, understandable language
            - If the answer cannot be found in the context, explicitly state "I do not know"

            **Objective**
            Provide a comprehensive yet easy-to-understand response that directly addresses the user's question while maintaining strict adherence to the given context.

            **Knowledge**
            - Only use information explicitly provided in the context
            - Do not add external information or speculation
            - Break down complex terms into layman's explanations
            - Maintain a clear, concise communication style

            **Constraints**
            - Absolutely do not fabricate or invent information
            - Start the answer immediately without preliminary remarks
            - Ensure each point is clear and directly related to the question
            - Translate technical jargon into simple English

            Context: {context}
            Question: {question}
            """

            with st.spinner("🤔 Thinking..."):
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                            temperature=0.7,
                            groq_api_key=os.environ["GROQ_API_KEY"],
                        ),
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )

                    response = qa_chain.invoke({'query': prompt})

                    result = response["result"]
                    source_documents = response["source_documents"]

                    with st.chat_message("assistant"):
                        st.markdown(result)
                        with st.expander("📚 View Source Documents"):
                            for i, doc in enumerate(source_documents):
                                content = doc.page_content
                                page_match = re.search(r"Page (\d+):", content)
                                if page_match:
                                    page_num = page_match.group(1)
                                    st.markdown(f"**Document {i+1} - Page {page_num}:**")
                                else:
                                    st.markdown(f"**Document {i+1}:**")
                                st.markdown(f"<div style='background:{card_bg}; padding:10px; border-radius:5px; margin:5px 0;'>{content[:300]}...</div>", unsafe_allow_html=True)

                    st.session_state.messages.append({'role': 'assistant', 'content': result})

                except Exception as e:
                    st.error(f"❌ Error processing your request: {str(e)}")


if __name__ == "__main__":
    main()
