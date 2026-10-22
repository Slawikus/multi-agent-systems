"""Agentic RAG demo: a stateful LangGraph agent with a single retrieval tool.

Flow:  agent -> (tool call?) -> retrieve -> grade -> agent -> ... -> END

* agent    decides whether it needs evidence; if so it calls the tool
* retrieve runs the tool and appends the result to the conversation
* grade    checks the evidence; sends the agent back to re-search, or gives up

Run `uv run python seed.py` once before this file.
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

MAX_RETRIES = 2


# --- 1. The tool: wraps a Chroma retriever so the model can call it -----------

retriever = Chroma(
    collection_name="architecture_records",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
).as_retriever(search_kwargs={"k": 3})


@tool
def search_architecture_records(query: str) -> str:
    """Search approved architecture records when evidence is needed."""
    return "\n\n".join(
        f'{doc.metadata["source"]}: {doc.page_content}'
        for doc in retriever.invoke(query)
    )


# --- 2. Models: one LLM, used two ways ----------------------------------------

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# the agent may emit tool calls
agent_model = model.bind_tools([search_architecture_records])


# the grader must answer with this exact shape
class Grade(BaseModel):
    sufficient: bool


grader_model = model.with_structured_output(Grade)


# --- 3. State: what survives between nodes ------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: nodes append, not replace
    retries: int  # how many times we sent the agent back to search
    status: str  # "answered" or "needs_review"


# --- 4. Nodes: plain functions that read state and return a partial update ----

SYSTEM = SystemMessage(
    "You are an assistant for an engineering team. For any question about how our "
    "systems are built, call search_architecture_records before answering and "
    "cite the ADR ids you used. Answer general questions directly."
)


def agent(state: AgentState) -> dict:
    response = agent_model.invoke([SYSTEM, *state["messages"]])
    return {"messages": [response], "status": "answered"}


# prebuilt node: executes the tool calls in the last message, appends ToolMessages
retrieve = ToolNode([search_architecture_records])


def grade(state: AgentState) -> dict:
    question = state["messages"][0].content
    evidence = state["messages"][-1].content
    verdict = grader_model.invoke(
        [
            SystemMessage("Is the evidence enough to answer the question?"),
            HumanMessage(f"Question: {question}\n\nEvidence:\n{evidence}"),
        ]
    )
    if verdict.sufficient:
        return {}
    if state["retries"] >= MAX_RETRIES:
        return {
            "status": "needs_review",
            "messages": [
                AIMessage("No architecture record covers this. Escalating to the architecture team.")
            ],
        }
    return {
        "retries": state["retries"] + 1,
        "messages": [
            HumanMessage("That evidence does not answer the question. Search again with different wording.")
        ],
    }


# --- 5. Edges: routing functions look at state and name the next node ---------

def after_agent(state: AgentState) -> str:
    return "retrieve" if state["messages"][-1].tool_calls else END


def after_grade(state: AgentState) -> str:
    return END if state["status"] == "needs_review" else "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", after_agent, ["retrieve", END])
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", after_grade, ["agent", END])

app = graph.compile()


# --- 6. Demo -------------------------------------------------------------------

def ask(question: str) -> None:
    result = app.invoke({"messages": [HumanMessage(question)], "retries": 0})
    print(f"\nQ: {question}")
    print(f"status: {result['status']}")
    print(f"A: {result['messages'][-1].content}")


if __name__ == "__main__":
    app.get_graph().print_ascii()
    # ask("Hi! What can you help me with?")
    ask("Where do we store customer-generated CSV exports, and for how long?")
    # ask("Which frontend framework do we use for the admin dashboard?")
