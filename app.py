import json
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict



RELEASE_DATA = {
    "2.4.0": {
        "tests": {
            "ok": True,
            "passed": 428,
            "failed": 0,
            "coverage": 86,
            "source": "ci://payments/2.4.0",
        },
        "operations": {
            "ok": True,
            "migration_check": "passed",
            "rollback_package": "available",
            "source": "deploy://payments/2.4.0",
        },
    },
    "2.4.1": {
        "tests": {
            "ok": True,
            "passed": 431,
            "failed": 0,
            "coverage": 87,
            "source": "ci://payments/2.4.1",
        },
        "operations": {
            "ok": False,
            "error": "deployment readiness service unavailable",
            "source": "deploy://payments/2.4.1",
        },
    },
}

@tool
def get_test_results(release_id: str) -> dict:
    """Return CI test evidence for a release."""
    return RELEASE_DATA[release_id]["tests"]


@tool
def get_operations_readiness(release_id: str) -> dict:
    """Return deployment-readiness evidence for a release."""
    return RELEASE_DATA[release_id]["operations"]


class SpecialistReport(BaseModel):
    specialist: Literal["testing", "operations"]
    status: Literal["ready", "not_ready", "unavailable"]
    summary: str
    evidence: list[str]
    source: str


class ReleaseDecision(BaseModel):
    decision: Literal["go", "no_go", "needs_human"]
    rationale: str
    blockers: list[str]
    sources: list[str]


MODEL = "openai:gpt-5-mini"

testing_agent = create_agent(
    model=MODEL,
    tools=[get_test_results],
    response_format=SpecialistReport,
    system_prompt=(
        "You are the testing specialist for a software release. "
        "Always call get_test_results for the supplied release ID. "
        "Use only the returned evidence. Set status to ready when no tests "
        "failed, not_ready when tests failed, and unavailable when the tool "
        "cannot provide evidence."
    ),
)

operations_agent = create_agent(
    model=MODEL,
    tools=[get_operations_readiness],
    response_format=SpecialistReport,
    system_prompt=(
        "You are the operations specialist for a software release. "
        "Always call get_operations_readiness for the supplied release ID. "
        "Use only the returned evidence. Set status to ready only when the "
        "migration check passed and a rollback package is available. Set "
        "status to unavailable when the evidence source reports an error."
    ),
)


release_coordinator = create_agent(
    model=MODEL,
    tools=[],
    response_format=ReleaseDecision,
    system_prompt=(
        "You are the release coordinator. Use only the specialist reports. "
        "Return go only when every specialist is ready. Return no_go when a "
        "specialist is not_ready. Return needs_human when required evidence "
        "is unavailable. Preserve the supplied source identifiers."
    ),
)


class ReleaseState(TypedDict):
    release_id: str
    reports: Annotated[list[dict], operator.add]
    final_decision: dict


def _run_specialist(agent, release_id: str) -> dict:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Assess release {release_id}.",
                }
            ]
        }
    )
    return result["structured_response"].model_dump()


def testing_node(state: ReleaseState) -> dict:
    report = _run_specialist(testing_agent, state["release_id"])
    return {"reports": [report]}


def operations_node(state: ReleaseState) -> dict:
    report = _run_specialist(operations_agent, state["release_id"])
    return {"reports": [report]}


def coordinator_node(state: ReleaseState) -> dict:
    ordered_reports = sorted(
        state["reports"],
        key=lambda report: report["specialist"],
    )

    result = release_coordinator.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Decide whether this release can proceed:\n\n"
                        + json.dumps(ordered_reports, indent=2)
                    ),
                }
            ]
        }
    )

    decision = result["structured_response"]
    return {"final_decision": decision.model_dump()}


builder = StateGraph(ReleaseState)

builder.add_node("testing_agent", testing_node)
builder.add_node("operations_agent", operations_node)
builder.add_node("release_coordinator", coordinator_node)

builder.add_edge(START, "testing_agent")
builder.add_edge(START, "operations_agent")
builder.add_edge(
    ["testing_agent", "operations_agent"],
    "release_coordinator",
)
builder.add_edge("release_coordinator", END)

graph = builder.compile()


def review_release(release_id: str) -> None:
    result = graph.invoke(
        {
            "release_id": release_id,
            "reports": [],
            "final_decision": {},
        }
    )

    print(f"\nRELEASE {release_id}")

    for report in sorted(
        result["reports"],
        key=lambda item: item["specialist"],
    ):
        print(
            f'{report["specialist"]}: '
            f'{report["status"]} — {report["summary"]}'
        )

    decision = result["final_decision"]
    print("decision:", decision["decision"])
    print("rationale:", decision["rationale"])
    print("sources:", decision["sources"])


review_release("2.4.0")
# review_release("2.4.1")
