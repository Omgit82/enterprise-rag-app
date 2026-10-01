import streamlit as st
import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="Enterprise Document AI", page_icon="📄", layout="wide")
st.title("📄 Enterprise Document AI")
st.markdown("Upload any PDF document and instantly ask questions. The AI will retrieve the exact context and provide grounded answers with citations.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
# Asking users for their own API key prevents your personal quota from being drained by the public
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
st.sidebar.markdown("[Get a free Gemini API key here](https://aistudio.google.com/)")

uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])

# Initialize in-memory tools (perfect for cloud deployment)
@st.cache_resource
def load_rag_tools():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Ephemeral client lives in memory and resets when the app restarts
    chroma_client = chromadb.Client() 
    return embedding_model, chroma_client

embedding_model, chroma_client = load_rag_tools()

# Session state to manage chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Document Ingestion
if uploaded_file and api_key:
    # Use the filename to create a unique collection for this session
    collection_name = "".join(e for e in uploaded_file.name if e.isalnum())
    
    if st.session_state.collection_name != collection_name:
        with st.sidebar.status("Processing Document..."):
            collection = chroma_client.create_collection(name=collection_name)
            reader = PdfReader(uploaded_file)
            
            chunk_id_counter = 0
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                chunk_size = 500
                overlap = 50
                for i in range(0, len(text), chunk_size - overlap):
                    chunk_text = text[i:i + chunk_size]
                    if len(chunk_text.strip()) < 30:
                        continue
                    
                    chunk_id = f"p{page_num}_{chunk_id_counter}"
                    chunk_id_counter += 1
                    
                    vector = embedding_model.encode(chunk_text).tolist()
                    collection.add(
                        documents=[chunk_text],
                        embeddings=[vector],
                        metadatas=[{"page": page_num + 1}],
                        ids=[chunk_id]
                    )
            st.session_state.collection_name = collection_name
        st.sidebar.success("Document Indexed Successfully!")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & RAG Execution
if prompt := st.chat_input("Ask a question about your document..."):
    if not api_key or not uploaded_file:
        st.error("Please upload a document and provide an API key in the sidebar.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                collection = chroma_client.get_collection(name=st.session_state.collection_name)
                q_vector = embedding_model.encode(prompt).tolist()
                
                results = collection.query(query_embeddings=[q_vector], n_results=3)
                retrieved_chunks = results['documents'][0]
                metadatas = results['metadatas'][0]
                
                context_blocks = []
                sources = set()
                for doc, meta in zip(retrieved_chunks, metadatas):
                    context_blocks.append(f"{doc}")
                    sources.add(f"Page {meta['page']}")
                    
                context_string = "\n\n".join(context_blocks)
                source_string = ", ".join(sources)
                
                system_prompt = f"""
                You are a precise document assistant. Answer the user's question using ONLY the provided document chunks below. 
                If the answer is not in the text, say "I couldn't find information about this in the provided document."
                
                QUESTION: {prompt}
                
                RETRIEVED CONTEXT:
                {context_string}
                """
                
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-1.5-flash', # Using the highly stable 1.5-flash model
                    contents=system_prompt,
                )
                
                full_response = f"{response.text}\n\n**Sources:** {source_string}"
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
