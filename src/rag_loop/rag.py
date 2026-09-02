import logging
from typing import TypedDict

import opik
from opik import opik_context
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .embeddings import get_embeddings

# We never bind tools to this chat model, so Google's automatic-function-calling (AFC) path
# never actually applies — silence its "not recommended" advisory, which the google-genai SDK
# logs unconditionally on first call regardless of whether tools are in use.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


class RAGState(TypedDict, total=False):
    question: str
    context: list[Document]
    answer: str


def _vector_store(settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(settings),
        collection_name=settings.collection_name,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


@opik.track(name="retrieve-context", type="tool")
def retrieve_context(question: str, settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()
    return _vector_store(settings).similarity_search(question, k=settings.retrieval_k)


@opik.track(name="generate-answer", type="llm")
def generate_answer(
    question: str, context: list[Document], settings: Settings | None = None
) -> str:
    settings = settings or get_settings()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an aircraft systems reference assistant. Answer only from the "
                "supplied context. If the context does not contain the answer, say you do "
                "not know rather than guessing at a procedure, limitation, or value. Cite "
                "page numbers when available.\n\n"
                "Context:\n{context}",
            ),
            ("human", "Question: {question}"),
        ]
    )
    chat_model = ChatGoogleGenerativeAI(
        model=settings.chat_model, api_key=settings.gemini_api_key, temperature=0
    )
    formatted_context = "\n\n".join(
        f"[page {document.metadata.get('page', '?') + 1}] {document.page_content}"
        for document in context
    )
    response = (prompt | chat_model).invoke(
        {"context": formatted_context, "question": question}
    )
    return response.text


def build_graph(settings: Settings | None = None):
    settings = settings or get_settings()

    def retrieve_node(state: RAGState) -> RAGState:
        return {"context": retrieve_context(state["question"], settings)}

    def answer_node(state: RAGState) -> RAGState:
        return {
            "answer": generate_answer(state["question"], state["context"], settings)
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


@opik.track(name="rag-loop", type="general")
def answer_question(question: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    result = build_graph(settings).invoke({"question": question})
    answer = {
        "question": question,
        "answer": result["answer"],
        "context": [document.page_content for document in result["context"]],
        "sources": [document.metadata for document in result["context"]],
    }
    opik_context.update_current_trace(
        input={"user_message": question},
        output={"assistant_response": answer["answer"]},
        metadata={
            "user_message": question,
            "assistant_response": answer["answer"],
            "retrieval_context": answer["context"],
            "sources": answer["sources"],
            "chat_model": settings.chat_model,
            "retrieval_k": settings.retrieval_k,
        },
        tags=[settings.trace_tag],
    )
    return answer
