
import os
from typing import List

from fastapi import UploadFile
from config.settings import (
    GOOGLE_API_KEY,
    VECTORSTORE_DIRECTORY,
    MODEL_OPTIONS,
)
from core.document_processor import (
    save_uploaded_file,
    load_documents_from_paths,
    split_documents_to_chunks,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils.logger import logger


def vectorstore_exists(persist_path: str) -> bool:
    """Check whether the vectorstore directory contains persisted data."""
    exists = os.path.exists(persist_path) and bool(os.listdir(persist_path))
    logger.debug(f"Vectorstore exists at {persist_path}: {exists}")
    return exists



def get_embeddings(model_provider: str):
    """Return the embedding model for the selected provider."""
    model_provider = model_provider.lower()
    logger.debug(f"Getting embeddings for provider: {model_provider}")

    if model_provider in ("groq", "ollama"):
        # Local embedding model; Groq and Ollama are LLM providers.
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )

    elif model_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

    else:
        logger.error(f"Unsupported LLM Provider: {model_provider}")
        raise ValueError(f"Unsupported LLM Provider: {model_provider}")

def initialize_empty_vectorstores():
    """Create a Chroma collection for each configured provider if needed."""
    logger.info("Initializing vectorstores...")

    for provider in MODEL_OPTIONS.keys():
        persist_path = VECTORSTORE_DIRECTORY[provider]
        os.makedirs(persist_path, exist_ok=True)

        if not vectorstore_exists(persist_path):
            embedding = get_embeddings(provider)

            Chroma(
                embedding_function=embedding,
                persist_directory=persist_path,
            )

            logger.debug(
                f"Initialized vectorstore for {provider} at {persist_path}"
            )
        else:
            logger.debug(
                f"Vectorstore directory already contains data: {persist_path}"
            )

    logger.info("Vectorstore initialization complete.")


async def upsert_vectorstore_from_pdfs(
    uploaded_files: List[UploadFile],
    model_provider: str,
):
    """Process uploaded PDFs and add their chunks to Chroma."""
    logger.debug(f"Upserting vectorstore for {model_provider}")

    if model_provider not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported LLM Provider: {model_provider}")

    if not uploaded_files:
        raise ValueError("No PDF files were provided.")

    file_paths = await save_uploaded_file(uploaded_files)

    if not file_paths:
        raise ValueError("No uploaded files were saved.")

    docs = load_documents_from_paths(file_paths)
    logger.info(f"Documents extracted: {len(docs)}")

    if not docs:
        raise ValueError(
            "No documents were extracted. Check the PDF files and document loader."
        )

    chunks = split_documents_to_chunks(docs)
    logger.info(f"Chunks generated: {len(chunks)}")

    # Remove chunks with empty or whitespace-only text.
    valid_chunks = [
        chunk
        for chunk in chunks
        if getattr(chunk, "page_content", None)
        and chunk.page_content.strip()
    ]

    if not valid_chunks:
        raise ValueError(
            "No valid text chunks were generated from the PDF. "
            "Check whether the PDF contains extractable text."
        )

    embedding = get_embeddings(model_provider)
    persist_path = VECTORSTORE_DIRECTORY[model_provider]
    os.makedirs(persist_path, exist_ok=True)

    # Open the existing collection, or create it if necessary.
    vectorstore = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding,
    )

    vectorstore.add_documents(valid_chunks)

    logger.info(
        f"Successfully added {len(valid_chunks)} chunks "
        f"to the vectorstore for {model_provider}."
    )

    return vectorstore


def load_vectorstore(model_provider: str):
    """Load the persisted vectorstore for a provider."""
    logger.debug(f"Loading vectorstore for provider: {model_provider}")

    if model_provider not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported LLM Provider: {model_provider}")

    persist_path = VECTORSTORE_DIRECTORY[model_provider]

    if not vectorstore_exists(persist_path):
        raise ValueError(
            f"Vectorstore directory not found or empty for provider: {model_provider}"
        )

    return Chroma(
        persist_directory=persist_path,
        embedding_function=get_embeddings(model_provider),
    )


def get_collections_count(model_provider: str) -> int:
    """Return the number of indexed chunks."""
    logger.debug(f"Getting collection count for provider: {model_provider}")

    vectorstore = load_vectorstore(model_provider)
    return vectorstore._collection.count()


def find_similar_chunks(model_provider: str, query: str):
    """Find chunks similar to the user's query."""
    logger.debug(f"Searching for similar chunks for provider: {model_provider}")

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    vectorstore = load_vectorstore(model_provider)

    if vectorstore._collection.count() == 0:
        raise ValueError(
            f"The vectorstore for {model_provider} is empty. Upload a PDF first."
        )

    return vectorstore.similarity_search(query)
