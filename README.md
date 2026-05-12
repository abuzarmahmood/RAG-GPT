# RAG-GPT

RAG-GPT is a Retrieval-Augmented Generation augmented LLM trained on a corpus of scientific articles relevant to the Katz Lab.
It is intended to be used both as a summarizer and synthesizer of knowledge as well as an LLM-augmented article search.

Currently, KatzGPT is deployed as a Streamlit Chatbot App.

Below are a couple examples of KatzGPT in action. 

## Quickstart

### Prerequisites

1. Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. Install required dependencies:
   ```bash
   pip install streamlit langchain openai chromadb pypdf tqdm joblib
   ```

### Setup

1. **Configure paths**: Edit `utils.py` to set your document paths:
   - `docs_path`: Directory containing your PDF files
   - `vector_persist_dir`: Directory where the vector store will be saved
   - `docs_output_dir`: Directory for intermediate document processing

2. **Generate vector store**: Run the vector store generation script to process your PDFs and create embeddings:
   ```bash
   python gen_vector_store.py
   ```
   This will:
   - Load all PDFs from your documents directory
   - Extract text and create document chunks
   - Generate embeddings using OpenAI
   - Store the vector database in the specified directory

3. **Launch the chatbot**: Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser to the URL shown (typically `http://localhost:8501`) and start asking questions!

## Examples

## Example 1
![image](https://github.com/abuzarmahmood/RAG-GPT/assets/12436309/42f36d81-4318-4673-94ce-1429394ecc1b)

## Example 2
![image](https://github.com/abuzarmahmood/RAG-GPT/assets/12436309/f97f79c2-ee30-4bff-bfc2-4d4bf58f35a6)
