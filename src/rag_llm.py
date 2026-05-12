"""
This file contains the functions for running the LLM.
"""

############################################################
## Imports 
############################################################

import os
from tqdm import tqdm
from pickle import dump, load
import numpy as np
from utils import return_paths, load_config

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

client = OpenAI()

############################################################
# Define Functions 
############################################################

def return_vectordb():
    """
    Return the vector database.

    Inputs:
        None

    Returns:
        vectordb: vector database
    """
    print("Loading vector database...")
    _, _, _, vector_persist_dir = return_paths()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=vector_persist_dir,
                      embedding_function=embeddings,
                      collection_metadata={"hnsw:space": "cosine"})
    print("Vector database loaded successfully")
    return vectordb


def get_docs(query, vectordb, k=5):
    """
    Get the documents for a given query.

    Inputs:
        query: query string
        k: number of documents to return

    Returns:
        docs_str: string of documents
    """

    outs = vectordb.similarity_search_with_relevance_scores(query, k=k)
    docs, scores = zip(*outs)

    docs_sources = list(set([doc.metadata['source'] for doc in docs]))

    docs_str = ""
    docs_str += '\n'
    docs_str += "=== Sources ===\n\n"
    for this_source, this_score in zip(docs_sources, scores):
        docs_str += f"{this_source.split('/')[-1]}"
        docs_str += f" === Score: {np.round(this_score,2)}\n\n"
    return docs_str


def run_query(
        query,
        vectordb,
        k=5
):
    """
    Run a query and return the answer.

    Inputs:
        query: query string
        k: number of documents to return

    Returns:
        out_str: string of answer
    """
    print(f"Searching for relevant documents (k={k})...")
    outs = vectordb.similarity_search_with_relevance_scores(query, k=k)
    docs, scores = zip(*outs)

    prompt = f"""
    Question: {query}
    =========
    {docs}
    =========
    Answer in Markdown:"""

    print(f"Generating response using {load_config()['model_name']}...")
    config = load_config()
    completion = client.chat.completions.create(model=config['model_name'],
    messages=[
        {"role": "system", "content": config['system_context']},
        {"role": "user", "content": prompt}
    ])

    out_str = completion.choices[0].message.content
    docs_str = get_docs(query, vectordb, k=k)
    out_str += '\n\n'
    out_str += docs_str
    return out_str
