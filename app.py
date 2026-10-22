from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

MAX_RETRIES = 2


class AgentState(TypedDict):
    messages: list
    """carries the conversation and model tool calls"""
    documents: list
    """stores evidence returned by Chroma"""
    retry_count: int
    """prevents an unlimited rewrite loop"""
    decision: str
    """names the next route"""
    answer: str
    """this together with sources and status describe outcome"""
    sources: list[str]
    status: str
    trace: list[str]
    """records which nodes actually ran"""


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="architecture_records",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


@tool
def search_architecture_records(query: str) -> str:
    """Search approved architecture records when evidence is needed."""
    documents = retriever.invoke(query)
    return "\n\n".join(
        f'{doc.metadata["source"]}: {doc.page_content}'
        for doc in documents
    )

decision_model = model.bind_tools([search_architecture_records])


def decide_or_respond(state: AgentState) -> dict:
    response = decision_model.invoke(state["messages"])
    return {
        "messages": [response],
        "trace": [*state["trace"], "decide"],
    }


def route_after_decision(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "retrieve"
    return "direct"


def route_after_assessment(state: AgentState) -> str:
    if state["decision"] == "answer":
        return "answer"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite"
    return "review"

graph = StateGraph(AgentState)

graph.add_node("decide_or_respond", decide_or_respond)
graph.add_node("retrieve", retrieve)
graph.add_node("assess", assess)
graph.add_node("rewrite", rewrite)
graph.add_node("answer", answer)
graph.add_node("review", review)

graph.add_edge(START, "decide_or_respond")
graph.add_conditional_edges(
    "decide_or_respond",
    route_after_decision,
    {"direct": END, "retrieve": "retrieve"},
)
graph.add_edge("retrieve", "assess")
graph.add_conditional_edges(
    "assess",
    route_after_assessment,
    {"answer": "answer", "rewrite": "rewrite", "review": "review"},
)
graph.add_edge("rewrite", "decide_or_respond")
graph.add_edge("answer", END)
graph.add_edge("review", END)

app = graph.compile()

if __name__ == "__main__":
    print(
        search_architecture_records.invoke(
            {"query": "customer-generated CSV export storage"}
        )
    )
