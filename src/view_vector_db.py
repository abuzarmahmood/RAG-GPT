"""
View documents in the vector database.
This script loads the vector database and displays information about stored documents.
"""

############################################################
## Imports 
############################################################

from utils import return_paths
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import Counter

############################################################
## Load Vector Database
############################################################

print("Loading vector database...")
_, _, _, vector_persist_dir = return_paths()
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=vector_persist_dir,
                  embedding_function=embeddings,
                  collection_metadata={"hnsw:space": "cosine"})

############################################################
## Display Database Information
############################################################

print("\n" + "="*60)
print("VECTOR DATABASE INFORMATION")
print("="*60 + "\n")

# Get all documents
data = vectordb.get()

if not data['ids']:
    print("No documents found in the vector database.")
else:
    num_docs = len(data['ids'])
    print(f"Total documents: {num_docs}\n")
    
    # Extract sources from metadata
    sources = []
    for metadata in data['metadatas']:
        if metadata and 'source' in metadata:
            source = metadata['source'].split('/')[-1]  # Get filename only
            sources.append(source)
    
    # Count documents per source
    source_counts = Counter(sources)
    
    print(f"Unique source files: {len(source_counts)}\n")
    print("Documents per source file:")
    print("-" * 60)
    
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} documents")
    
    print("\n" + "="*60)
    
    # Display sample document IDs
    print("\nSample document IDs (first 10):")
    print("-" * 60)
    for doc_id in data['ids'][:10]:
        print(f"  {doc_id}")
    
    if num_docs > 10:
        print(f"  ... and {num_docs - 10} more")
    
    print("\n" + "="*60)

print("\nDone!")
