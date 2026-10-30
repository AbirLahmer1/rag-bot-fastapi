
from config.settings import GROQ_API_KEY, GOOGLE_API_KEY

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from utils.logger import logger


def get_prompt():
    logger.debug("Creating chat prompt template.")

    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the user's question using the provided context. "
            "If the context does not contain the answer, say "
            "'I don't know based on the provided document.' "
            "Answer in the same language as the user's question."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{input}"
        ),
    ])


def get_llm(model_provider: str, model: str):
    logger.debug(f"Initializing LLM for {model_provider} - {model}")

    if model_provider.lower() == "groq":
        return ChatGroq(
            model=model,
            api_key=GROQ_API_KEY,
        )

    elif model_provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
        )

    elif model_provider.lower() == "ollama":
        return ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0,
        )

    else:
        logger.error(f"Unsupported LLM Provider: {model_provider}")
        raise ValueError(
            f"Unsupported LLM Provider: {model_provider}"
        )


def build_llm_chain(model_provider: str, model: str, vectorstore):
    logger.debug(
        f"Building LLM chain for provider: {model_provider}, model: {model}"
    )

    prompt = get_prompt()
    llm = get_llm(model_provider, model)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt=prompt,
    )

    return create_retrieval_chain(
        retriever,
        document_chain,
    )
