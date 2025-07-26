# 📦 Import all required libraries
import streamlit as st  # Streamlit for building UI
from PyPDF2 import PdfReader  # To extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For breaking text into chunks
from langchain_community.vectorstores import FAISS  # FAISS for vector similarity search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face FLAN-T5 model and tokenizer
from huggingface_hub import login  # Login for Hugging Face model access
from langchain_huggingface import HuggingFaceEmbeddings  # To generate sentence embeddings
import torch
import warnings

# 🚫 Ignore unnecessary warnings for cleaner output
warnings.filterwarnings("ignore")

# 🔐 Login to Hugging Face using your access token
login(st.secrets["huggingface_token"])

# 🎨 Streamlit UI setup
st.set_page_config(page_title="📄 Chat with your PDF")
st.header("Free Chatbot (FLAN-T5 + PDF Reader)")

# 📤 Sidebar for file upload
with st.sidebar:
    st.title("Upload your document")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# 🧠 Load the embedding model (Sentence Transformers)
@st.cache_resource
def load_embedder():
    # Using Hugging Face's MiniLM model to generate embeddings for each text chunk
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_function = load_embedder()

# 🤖 Load FLAN-T5 model for answering questions
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# 📄 Process the uploaded PDF
if file is not None:
    with st.spinner("📄 Reading PDF..."):
        # Read the PDF and extract text from each page
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    # ⚠️ If no text was extracted, show warning
    if not text.strip():
        st.warning("No text found in PDF.")
    else:
        st.success("✅ PDF loaded successfully.")

        # 🔪 Split text into manageable chunks for embedding & retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,     # max 500 characters per chunk
            chunk_overlap=100,  # 100-character overlap between chunks
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 🧠 Create vector store using FAISS
        with st.spinner("🔍 Creating vector index..."):
            vector_store = FAISS.from_texts(chunks, embedding_function)

        # 🧾 Get question input from user
        question = st.text_input("Ask something from the PDF")

        # ✅ When user submits a question
        if question:
            with st.spinner("🤖 Generating answer..."):
                # 🔍 Find the most relevant chunk based on similarity
                docs_and_scores = vector_store.similarity_search_with_score(question, k=1)
                best_chunk = docs_and_scores[0][0].page_content

                # 📜 Prepare the prompt for the FLAN-T5 model
                prompt = f"""You are a helpful assistant. Based on the context below, answer the question clearly and completely in bullet points. 
If the question asks about types of installation media, list all the types like USB flash drive, DVD, and ISO file.

Context:
{best_chunk}

Question: {question}
Answer:"""

                # ✏️ Tokenize input and generate output from model
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,              # Controls max response length
                    temperature=0.7,                 # Controls randomness (0 = deterministic)
                    do_sample=True,
                    top_p=0.9,                       # Top-p sampling for creative answers
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                # 📤 Decode model output into human-readable answer
                raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 🧼 Clean up the answer (remove extra prompt echo if any)
                if "Answer:" in raw_answer:
                    final_answer = raw_answer.split("Answer:")[-1].strip()
                else:
                    final_answer = raw_answer.strip()

                # 📋 Convert response into a list of bullet points
                lines = final_answer.split("\n")
                bullet_lines = [f"- {line.strip()}" for line in lines if line.strip()]

                # 💬 Display the final answer to the user
                st.markdown("### 💬 Answer:")
                for bullet in bullet_lines:
                    st.markdown(bullet)
