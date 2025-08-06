# StudyBuddy

StudyBuddy is an AI-powered assistant that reads PDF study materials, breaks them into context-aware chunks, and enables fast and accurate question answering using a local LLaMA model.

## Features
Chunking of PDF documents into overlapping text segments
Embedding and vector storage for fast retrieval
Retrieval-Augmented Generation (RAG) using LLaMA
Local model execution (via llama.cpp)

## Directory Structure
.
├── scripts/
│   ├── chunker.py           PDF loading and chunking logic  
│   ├── embedder.py          Embedding and vector store management  
│   ├── retriever.py         Search logic for context retrieval  
│   ├── generate.py          Query to LLaMA based on retrieved context  
│   └── ui.py                Simple ui for users to interact with  
├── data/                    Folder to store PDFs and processed data  
├── models/                  (Optional) local LLaMA model weights path  
├── README.md  
└── requirements.txt  

## Installation
git clone https://github.com/kuro-spirit/StudyBuddy.git  
cd studybuddy  
python -m venv venv  
.\venv\Scripts\activate  
pip install -r requirements.txt  

## Usage
### Step 1: Run ui file
python ui.py
### Step 2: Upload study resource of pdf type
Click upload button and select pdf you want to study
### Step 3: Ask a question
Input question into chat box and press ask for llama to process question

## Architecture Overview
PDF ➝ Chunker ➝ Embedder ➝ Vector Store ➝ Retriever ➝ LLaMA Prompt ➝ Answer
