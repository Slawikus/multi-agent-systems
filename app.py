import operator
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

MAX_RETRIES = 2


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
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
    trace: Annotated[list[str], operator.add]
    """records which nodes actually ran"""


class Assessment(BaseModel):
    decision: Literal["answer", "rewrite"]
    reason: str


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
assessment_model = model.with_structured_output(Assessment)

DECIDE_PROMPT = SystemMessage(
    "You are an assistant for an engineering team. Answer small talk or general "
    "questions directly. For any question about how our systems are built, you "
    "must call search_architecture_records before answering."
)


def question_of(state: AgentState) -> str:
    """The original user question is always the first message."""
    return state["messages"][0].content


def evidence_of(state: AgentState) -> str:
    return "\n\n".join(
        f'{doc.metadata["source"]}: {doc.page_content}'
        for doc in state["documents"]
    )


def decide_or_respond(state: AgentState) -> dict:
    response = decision_model.invoke([DECIDE_PROMPT, *state["messages"]])
    update = {"messages": [response], "trace": ["decide"]}
    if not response.tool_calls:
        update |= {"answer": response.content, "status": "answered_directly"}
    return update


def retrieve(state: AgentState) -> dict:
    """Run the model's tool calls against Chroma and feed results back."""
    tool_messages, documents = [], []
    for call in state["messages"][-1].tool_calls:
        docs = retriever.invoke(call["args"]["query"])
        documents.extend(docs)
        tool_messages.append(
            ToolMessage(
                content="\n\n".join(
                    f'{d.metadata["source"]}: {d.page_content}' for d in docs
                ),
                tool_call_id=call["id"],
            )
        )
    return {
        "messages": tool_messages,
        "documents": documents,
        "sources": sorted({d.metadata["source"] for d in documents}),
        "trace": ["retrieve"],
    }


def assess(state: AgentState) -> dict:
    """Ask the model whether the evidence actually answers the question."""
    verdict = assessment_model.invoke(
        [
            SystemMessage(
                "Decide whether the evidence is enough to answer the question. "
                "Reply 'answer' if it is, 'rewrite' if a different search is needed."
            ),
            HumanMessage(
                f"Question: {question_of(state)}\n\nEvidence:\n{evidence_of(state)}"
            ),
        ]
    )
    return {"decision": verdict.decision, "trace": [f"assess:{verdict.decision}"]}


def rewrite(state: AgentState) -> dict:
    """Send the model back to search again with a different query."""
    return {
        "messages": [
            HumanMessage(
                "The retrieved records did not answer the question. Call "
                "search_architecture_records again with a differently phrased query."
            )
        ],
        "retry_count": state["retry_count"] + 1,
        "trace": ["rewrite"],
    }


def answer(state: AgentState) -> dict:
    """Write the final answer grounded only in the retrieved records."""
    response = model.invoke(
        [
            SystemMessage(
                "Answer the question using only the evidence. Cite the ADR ids you used."
            ),
            HumanMessage(
                f"Question: {question_of(state)}\n\nEvidence:\n{evidence_of(state)}"
            ),
        ]
    )
    return {"answer": response.content, "status": "answered", "trace": ["answer"]}


def review(state: AgentState) -> dict:
    """Give up after MAX_RETRIES and hand the question to a human."""
    return {
        "answer": (
            f"No architecture record covers this after {state['retry_count']} "
            "retries. Escalated to the architecture team for review."
        ),
        "status": "needs_review",
        "trace": ["review"],
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

def ask(question: str) -> None:
    result = app.invoke(
        {"messages": [HumanMessage(question)], "retry_count": 0, "documents": []}
    )
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"status: {result['status']}")
    print(f"sources: {result.get('sources', [])}")
    print(f"trace: {' -> '.join(result['trace'])}")


if __name__ == "__main__":
    ask("Hi! What can you help me with?")
    ask("Where do we store customer-generated CSV exports, and for how long?")
    ask("Which frontend framework do we use for the admin dashboard?")
