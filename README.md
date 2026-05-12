# RAG-GPT

RAG-GPT is a Retrieval-Augmented Generation augmented LLM trained on a corpus of scientific articles. 
It is intended to be used both as a summarizer and synthesizer of knowledge as well as an LLM-augmented article search.

Currently, RAG-GPT is deployed as a Streamlit Chatbot App.

The [Examples](#examples) section illustrates outputs of RAG-GPT. 

## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd RAG-GPT
```

### Step 2: Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install packages individually:
```bash
pip install streamlit langchain openai chromadb pypdf tqdm joblib numpy
```

### Step 4: Configure OpenAI API Key

Set up your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-api-key-here
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

**Note:** For persistent configuration, add the export command to your `~/.bashrc` or `~/.zshrc` file (Linux/Mac) or set it as a system environment variable (Windows).

## Quickstart

### Setup

1. **Configure paths**: Copy and edit `config.json` to set your document paths:
   ```bash
   cp config.json config.json
   ```
   Then edit the file to set:
   - `docs_path`: Directory containing your PDF files
   - `vector_persist_dir`: Directory where the vector store will be saved
   - `docs_output_dir`: Directory for intermediate document processing

2. **Generate vector store**: Run the vector store generation script to process your PDFs and create embeddings:
   ```bash
   python src/gen_vector_store.py
   ```
   This will:
   - Load all PDFs from your documents directory
   - Extract text and create document chunks
   - Generate embeddings using OpenAI
   - Store the vector database in the specified directory

3. **Launch the chatbot**: Start the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

4. Open your browser to the URL shown (typically `http://localhost:8501`) and start asking questions!

## Examples

## Example 1
![image](https://github.com/abuzarmahmood/RAG-GPT/assets/12436309/42f36d81-4318-4673-94ce-1429394ecc1b)

## Example 2
![image](https://github.com/abuzarmahmood/RAG-GPT/assets/12436309/f97f79c2-ee30-4bff-bfc2-4d4bf58f35a6)
