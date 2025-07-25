#!/usr/bin/env python3
"""
"""

import os
import re
import asyncio
import threading
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Core imports with error handling
try:
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from pydantic import BaseModel
    import PyPDF2
    from langdetect import detect
    import chromadb
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install fastapi uvicorn gradio google-generativeai chromadb sentence-transformers PyPDF2 langdetect python-multipart numpy pydantic")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================

class Config:
    GEMINI_API_KEY = "....."  # Replace with your key
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MAX_RETRIEVED_CHUNKS = 5
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Simpler, more reliable model
    VECTOR_DB_PATH = "./chroma_db"
    UPLOAD_DIR = "./uploads"

# Initialize Gemini
try:
    if Config.GEMINI_API_KEY and Config.GEMINI_API_KEY != "your-api-key-here":
        genai.configure(api_key=Config.GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    else:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured from environment")
        else:
            logger.error("No Gemini API key found")
            raise ValueError("Please set GEMINI_API_KEY")
except Exception as e:
    logger.error(f"Gemini configuration failed: {e}")
    genai = None

# ================================
# MODELS
# ================================

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]
    language: str
    confidence_score: float

# ================================
# DOCUMENT PROCESSOR
# ================================

class DocumentProcessor:
    def __init__(self):
        self.model = None
        if genai:
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk = ' '.join(words[i:i + Config.CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

# ================================
# VECTOR STORE
# ================================

class VectorStore:
    def __init__(self):
        try:
            os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
            self.collection = self.client.get_or_create_collection(name="documents")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            raise e
    
    def add_documents(self, chunks: List[str]):
        if not chunks:
            return
        
        try:
            embeddings = self.embedding_model.encode(chunks)
            ids = [f"chunk_{i}_{int(datetime.now().timestamp())}" for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=ids
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def search(self, query: str, n_results: int = Config.MAX_RETRIEVED_CHUNKS) -> List[str]:
        if not query:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            if results['documents'] and results['documents'][0]:
                return results['documents'][0]
            return []
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ================================
# RAG PIPELINE
# ================================

class RAGPipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.conversation_history = []
        
        # Initialize Gemini model
        self.model = None
        if genai:
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("RAG Pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
    
    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return 'bengali' if lang == 'bn' else 'english'
        except:
            return 'english'
    
    def process_document(self, pdf_path: str):
        logger.info(f"Processing document: {pdf_path}")
        
        # Extract and chunk text
        raw_text = self.doc_processor.extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise ValueError("Could not extract text from PDF")
        
        chunks = self.doc_processor.chunk_text(raw_text)
        if not chunks:
            raise ValueError("No chunks created from document")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        logger.info(f"Successfully processed document with {len(chunks)} chunks")
    
    def generate_answer(self, query: str, context: List[str], language: str) -> str:
        if not self.model:
            return "AI model not available. Please check configuration."
        
        try:
            context_text = "\n\n".join(context[:3])  # Use top 3 chunks
            
            if language == 'bengali':
                prompt = f"""
                আপনি একটি সহায়ক AI সহকারী। নিচের তথ্যের ভিত্তিতে প্রশ্নের উত্তর দিন।
                
                প্রাসঙ্গিক তথ্য:
                {context_text}
                
                প্রশ্ন: {query}
                
                স্পষ্ট এবং সঠিক উত্তর দিন। যদি তথ্য না থাকে, তা বলুন।
                """
            else:
                prompt = f"""
                You are a helpful AI assistant. Answer the question based on the provided information.
                
                Relevant information:
                {context_text}
                
                Question: {query}
                
                Provide a clear and accurate answer. If the information is not available, say so.
                """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            error_msg = "দুঃখিত, উত্তর তৈরি করতে সমস্যা হচ্ছে।" if language == 'bengali' else "Sorry, I couldn't generate an answer."
            return error_msg
    
    def query(self, query: str) -> QueryResponse:
        if not query or not query.strip():
            return QueryResponse(
                answer="Please provide a valid question.",
                retrieved_chunks=[],
                language="english",
                confidence_score=0.0
            )
        
        try:
            # Detect language and search
            language = self.detect_language(query)
            retrieved_chunks = self.vector_store.search(query)
            
            if not retrieved_chunks:
                no_info_msg = "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।" if language == 'bengali' else "No relevant information found."
                return QueryResponse(
                    answer=no_info_msg,
                    retrieved_chunks=[],
                    language=language,
                    confidence_score=0.0
                )
            
            # Generate answer
            answer = self.generate_answer(query, retrieved_chunks, language)
            
            # Simple confidence score
            confidence = min(len(retrieved_chunks) * 0.2, 1.0)
            
            # Store in history
            self.conversation_history.append({'query': query, 'answer': answer})
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)
            
            return QueryResponse(
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                language=language,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResponse(
                answer="An error occurred while processing your question.",
                retrieved_chunks=[],
                language="english",
                confidence_score=0.0
            )

# ================================
# INITIALIZE SYSTEM
# ================================

# Create directories
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)

# Initialize RAG pipeline
try:
    rag_pipeline = RAGPipeline()
    logger.info("System initialized successfully")
except Exception as e:
    logger.error(f"System initialization failed: {e}")
    rag_pipeline = None

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(title="Multilingual RAG API", version="2.1")

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        rag_pipeline.process_document(file_path)
        return {"message": f"Document {file.filename} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        return rag_pipeline.query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if rag_pipeline else "error",
        "timestamp": datetime.now().isoformat()
    }

# ================================
# GRADIO INTERFACE (FIXED)
# ================================

def create_gradio_interface():
    """Create a properly working Gradio interface"""
    
    def process_message(message, history):
        """Process chat message"""
        if not message or not message.strip():
            return history, ""
        
        if not rag_pipeline:
            history.append([message, "❌ System not initialized"])
            return history, ""
        
        try:
            response = rag_pipeline.query(message.strip())
            formatted_answer = f"{response.answer}\n\n📊 Confidence: {response.confidence_score:.2f}"
            history.append([message, formatted_answer])
            return history, ""
        except Exception as e:
            history.append([message, f"❌ Error: {str(e)}"])
            return history, ""
    
    def upload_file(file):
        """Process uploaded file"""
        if not file:
            return "❌ No file selected"
        
        if not rag_pipeline:
            return "❌ System not initialized"
        
        try:
            rag_pipeline.process_document(file.name)
            return f"✅ Success: {os.path.basename(file.name)} processed!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_history():
        """Clear chat history"""
        if rag_pipeline:
            rag_pipeline.conversation_history = []
        return []
    
    # Create interface
    with gr.Blocks(title="Multilingual RAG System") as demo:
        gr.Markdown("# 🤖 Multilingual RAG System")
        gr.Markdown("Upload PDF documents and ask questions in English or Bengali!")
        
        with gr.Tabs():
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(
                    type="tuples",  # Fix the deprecation warning
                    height=400,
                    label="Chat with your documents"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question...",
                        scale=4,
                        container=False
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                
                clear = gr.Button("Clear Chat")
                
                gr.Examples(
                    examples=[
                        "What is this document about?",
                        "এই নথিতে কী আছে?",
                        "Summarize the main points",
                        "মূল বিষয়গুলো বলুন"
                    ],
                    inputs=msg
                )
            
            with gr.TabItem("📄 Upload"):
                gr.Markdown("### Upload PDF Document")
                
                file_input = gr.File(
                    label="Select PDF",
                    file_types=[".pdf"]
                )
                upload_btn = gr.Button("Process Document", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
        
        # Connect events
        msg.submit(process_message, [msg, chatbot], [chatbot, msg])
        send.click(process_message, [msg, chatbot], [chatbot, msg])
        clear.click(clear_history, outputs=[chatbot])
        upload_btn.click(upload_file, [file_input], [status])
    
    return demo

# ================================
# SERVER FUNCTIONS
# ================================

def run_api_server(port: int = 8000):
    """Run FastAPI server"""
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"API server failed: {e}")

def run_gradio_server(port: int = 7860):
    """Run Gradio server with proper settings"""
    try:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False
        )
    except Exception as e:
        logger.error(f"Gradio server failed: {e}")
        print(f"❌ Gradio failed: {e}")

# ================================
# MAIN FUNCTION
# ================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual RAG System")
    parser.add_argument("--mode", choices=["api", "gradio", "all"], default="all")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--gradio-port", type=int, default=7860)
    
    args = parser.parse_args()
    
    if not rag_pipeline:
        print("❌ System initialization failed!")
        print("Please check your Gemini API key and dependencies")
        return
    
    print("🤖 Multilingual RAG System v2.1")
    print("✅ System ready")
    
    if args.mode == "api":
        print(f"🚀 API server: http://localhost:{args.api_port}")
        run_api_server(args.api_port)
    elif args.mode == "gradio":
        print(f"🎨 Web interface: http://localhost:{args.gradio_port}")
        run_gradio_server(args.gradio_port)
    else:
        print(f"🚀 API: http://localhost:{args.api_port}")
        print(f"🎨 Web: http://localhost:{args.gradio_port}")
        
        # Start both servers
        api_thread = threading.Thread(target=lambda: run_api_server(args.api_port), daemon=True)
        api_thread.start()
        time.sleep(2)
        run_gradio_server(args.gradio_port)

if __name__ == "__main__":
    main()
