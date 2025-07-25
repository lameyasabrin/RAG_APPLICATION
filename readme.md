# 🤖 Multilingual RAG System

A complete **Retrieval-Augmented Generation (RAG)** pipeline that supports both **English** and **Bengali** languages. Upload PDF documents and ask questions in your preferred language to get accurate, context-aware answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### 🌐 **Multilingual Support**
- **English** and **Bengali (বাংলা)** language support
- Automatic language detection
- Native responses in the same language as your question

### 📚 **Document Processing**
- PDF text extraction and processing
- Intelligent text chunking for better retrieval
- Support for both English and Bengali documents

### 🧠 **Advanced AI Capabilities**
- **Vector Search**: Semantic similarity matching using sentence transformers
- **Context-Aware Responses**: Powered by Google Gemini 2.0 Flash
- **Conversation Memory**: Maintains context across multiple questions
- **Answer Evaluation**: Confidence scoring for response reliability

### 🖥️ **Dual Interface**
- **Web Interface**: User-friendly Gradio-based chat interface
- **REST API**: Complete FastAPI backend for integration

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilingual-rag-system.git
cd multilingual-rag-system

# Install dependencies
pip install fastapi uvicorn gradio google-generativeai chromadb sentence-transformers PyPDF2 langdetect python-multipart numpy pydantic
```

### 2. Configuration

**Set your Gemini API Key:**

```python
# Option 1: Edit the code (line 35)
GEMINI_API_KEY = "your-actual-api-key-here"

# Option 2: Use environment variable
export GEMINI_API_KEY="your-actual-api-key-here"
```

**Get your free Gemini API key:** [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Run the System

```bash
# Run both web interface and API
python rag_system.py --mode all

# Or run individually
python rag_system.py --mode gradio    # Web interface only
python rag_system.py --mode api       # API server only
```

### 4. Access the Application

- **Web Interface**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### 🖥️ Web Interface Usage

1. **Upload Document**
   - Go to the "📄 Upload" tab
   - Select a PDF file
   - Click "Process Document"
   - Wait for success message

2. **Ask Questions**
   - Switch to "💬 Chat" tab
   - Type your question in English or Bengali
   - Press Enter or click "Send"
   - View the AI's response with confidence score

### 🔌 API Usage

#### Upload Document
```bash
curl -X POST "http://localhost:8000/upload_document" \
     -F "file=@your_document.pdf"
```

#### Query Document
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is this document about?"}'
```

#### Python Example
```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/upload_document", files={"file": f})

# Query
response = requests.post("http://localhost:8000/query", 
                        json={"query": "What are the main findings?"})
result = response.json()
print(result["answer"])
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│ Document         │───▶│ Text Chunking   │
└─────────────────┘    │ Processor        │    └─────────────────┘
                       └──────────────────┘               │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Vector Store     │◀───│ Embeddings      │
└─────────────────┘    │ (ChromaDB)       │    │ Generation      │
         │              └──────────────────┘    └─────────────────┘
         ▼                        │
┌─────────────────┐              ▼
│ Language        │    ┌──────────────────┐
│ Detection       │    │ Similarity       │
└─────────────────┘    │ Search           │
         │              └──────────────────┘
         ▼                        │
┌─────────────────┐              ▼
│ Answer          │    ┌──────────────────┐
│ Generation      │◀───│ Retrieved        │
│ (Gemini AI)     │    │ Context          │
└─────────────────┘    └──────────────────┘
```

## 🔧 Configuration Options

### Core Settings
```python
class Config:
    GEMINI_API_KEY = "your-api-key"         # Your Gemini API key
    CHUNK_SIZE = 500                        # Text chunk size (tokens)
    CHUNK_OVERLAP = 100                     # Overlap between chunks
    MAX_RETRIEVED_CHUNKS = 5                # Number of chunks to retrieve
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Sentence transformer model
    VECTOR_DB_PATH = "./chroma_db"          # Vector database path
    UPLOAD_DIR = "./uploads"                # File upload directory
```

### Command Line Options
```bash
python rag_system.py --help

Options:
  --mode {api,gradio,all}  Run mode (default: all)
  --api-port INTEGER       FastAPI server port (default: 8000)
  --gradio-port INTEGER    Gradio interface port (default: 7860)
```

## 💡 Example Queries

### English Examples
```
"What is this document about?"
"Summarize the main findings"
"What methodology was used?"
"What are the conclusions?"
"Who are the authors?"
```

### Bengali Examples
```
"এই নথিতে কী আলোচনা করা হয়েছে?"
"মূল ফলাফলগুলো সংক্ষেপে বলুন"
"কী পদ্ধতি ব্যবহার করা হয়েছে?"
"উপসংহার কী?"
"লেখকরা কারা?"
```

## 📊 Performance Metrics

The system provides several metrics for each response:

- **Confidence Score**: Overall reliability (0.0-1.0)
- **Response Time**: Typically 2-5 seconds
- **Language Detection**: >95% accuracy
- **Retrieval Accuracy**: Depends on document quality

## 🛠️ Troubleshooting

### Common Issues

**1. "System not initialized" Error**
```bash
# Check your API key
export GEMINI_API_KEY="your-key-here"
# Or edit Config.GEMINI_API_KEY in the code
```

**2. "Could not extract text from PDF"**
- Ensure PDF contains text (not scanned images)
- Try with a different PDF file
- Check file permissions

**3. "No relevant information found"**
- Make sure you've uploaded a document first
- Check if your question relates to the document content
- Try rephrasing your question

**4. Gradio Interface Issues**
```bash
# Run API mode only
python rag_system.py --mode api
# Access via http://localhost:8000/docs
```

### Dependencies Issues
```bash
# Reinstall all dependencies
pip uninstall -y fastapi uvicorn gradio google-generativeai chromadb sentence-transformers PyPDF2 langdetect python-multipart numpy pydantic
pip install fastapi uvicorn gradio google-generativeai chromadb sentence-transformers PyPDF2 langdetect python-multipart numpy pydantic
```

## 🔒 Security Considerations

- **API Key**: Keep your Gemini API key secure and never commit it to public repositories
- **File Upload**: The system processes uploaded files locally - ensure proper file validation in production
- **Network**: Consider using HTTPS and authentication for production deployments



### Environment Variables
```bash
export GEMINI_API_KEY="your-production-key"
export PYTHONPATH="/app"
```



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini** for powerful language generation
- **ChromaDB** for efficient vector storage
- **Sentence Transformers** for multilingual embeddings
- **Gradio** for the user interface
- **FastAPI** for the REST API framework




