"""
General utility functions for PDF loading, text extraction, and mathematical calculations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import fitz  # PyMuPDF


def extract_text_from_pdf(file_stream: Any) -> str:
    """
    Extracts all raw text content page-by-page from an uploaded PDF file stream.

    Args:
        file_stream: A file-like object containing the PDF binary data.

    Returns:
        Concatenated text extracted from the document.
    """
    text = ""
    pdf = fitz.open(stream=file_stream.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes the cosine similarity metric between two dense vectors.

    Args:
        vec1: List of floats representing vector A.
        vec2: List of floats representing vector B.

    Returns:
        Cosine similarity score ranging from -1.0 to 1.0.
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def find_similar_papers(uploaded_embedding: List[float], dataset: List[Dict[str, Any]], top_n: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Finds the top N semantically similar research papers in the database based on cosine similarity of embeddings.

    Args:
        uploaded_embedding: Dense embedding vector of the uploaded paper's abstract.
        dataset: Database list containing candidate papers and their precomputed embeddings.
        top_n: Number of similar recommendations to return.

    Returns:
        Sorted list of tuples containing (similarity_score, paper_record).
    """
    matches = []
    for paper in dataset:
        candidate_embedding = paper.get("text_embedding")
        if candidate_embedding is not None:
            score = calculate_cosine_similarity(uploaded_embedding, candidate_embedding)
            matches.append((score, paper))
            
    # Sort descending by similarity score
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[:top_n]
