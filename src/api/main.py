import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.local_inference.phi_pipeline import PhiNPUPipeline
from src.remote_inference.remote_client import get_remote_client
from src.classifier.rule_based import RuleBasedClassifier
from src.classifier.llm_judged import LLMJudgedClassifier
from src.router.router import RoutingAgent, ClassifierType

load_dotenv()

# ---- shared state, populated on startup ----
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading NPU pipeline...")
    pipeline = PhiNPUPipeline()
    pipeline.load()

    remote_client = get_remote_client()
    rule_clf = RuleBasedClassifier()
    llm_clf = LLMJudgedClassifier(pipeline)

    state["agent"] = RoutingAgent(
        local_pipeline=pipeline,
        remote_client=remote_client,
        rule_based_classifier=rule_clf,
        llm_judged_classifier=llm_clf,
        default_classifier=ClassifierType.RULE_BASED,
    )
    state["pipeline_ready"] = True
    print("[Startup] Ready.")

    yield

    print("[Shutdown] Cleaning up.")
    state.clear()


app = FastAPI(
    title="Hybrid Token-Efficient Routing Agent",
    description="Routes queries between local NPU inference and remote LLM inference.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- request/response schemas ----

class ConstraintsModel(BaseModel):
    max_latency_ms: Optional[int] = None
    prefer_local: bool = False


class MetadataModel(BaseModel):
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    context: list[dict] = []
    constraints: ConstraintsModel = ConstraintsModel()
    metadata: MetadataModel = MetadataModel()
    classifier: Optional[str] = None  # "rule_based" | "llm_judged"


class RouteResponse(BaseModel):
    response: str
    route_taken: str
    classifier_used: str
    classifier_signals: dict
    classifier_latency_ms: int
    fallback_triggered: bool
    tokens: dict
    latency_ms: int
    model_used: str


# ---- endpoints ----

@app.get("/health")
def health():
    return {
        "status": "ok" if state.get("pipeline_ready") else "loading",
        "npu_ready": state.get("pipeline_ready", False),
    }


@app.post("/route", response_model=RouteResponse)
def route_query(req: QueryRequest):
    if not state.get("pipeline_ready"):
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    agent: RoutingAgent = state["agent"]

    classifier_enum = None
    if req.classifier:
        try:
            classifier_enum = ClassifierType(req.classifier)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid classifier '{req.classifier}'. Use 'rule_based' or 'llm_judged'.",
            )

    try:
        result = agent.route(
            query=req.query,
            context=req.context,
            classifier=classifier_enum,
            prefer_local=req.constraints.prefer_local,
            max_latency_ms=req.constraints.max_latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result