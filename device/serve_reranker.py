#!/usr/bin/env python3
"""OpenAI-style rerank service (FastAPI) for the QNN Qwen3-Reranker, Jina/Cohere compatible.

OpenAI itself defines no rerank endpoint; the de-facto standard used by vLLM, Xinference, LocalAI,
Dify, FastGPT, LangChain, ... is the Jina/Cohere format implemented here:

  POST /v1/rerank   (also /rerank, /v2/rerank)
       {"model": "...", "query": "...", "documents": ["..." | {"text": "..."}],
        "top_n": 3, "return_documents": true, "instruction": "(optional task instruction)"}
    -> {"id": "rerank-...", "object": "rerank", "model": "...", "usage": {"total_tokens": N},
        "results": [{"index": 0, "relevance_score": 0.98, "margin": 7.6, "document": {"text": "..."}}]}
  GET  /v1/models   OpenAI model list
  GET  /health      200 once the model is loaded
  GET  /docs        interactive API docs

relevance_score = P(yes) in [0, 1] (as in the official usage); results are ordered by the logit margin
(logit(yes) - logit(no)), which stays distinct where fp16 P(yes) saturates near 1.

The NPU executes one query-document pair at a time. Requests are handled asynchronously and their
NPU work is queued on a single worker thread, so the event loop never blocks. Run exactly one server
process: every process loads its own copy of the model into NPU memory.

  source setup_env.sh
  python3 serve_reranker.py --host 0.0.0.0 --port 8000 [--api-key sk-...]
"""
import argparse
import asyncio
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---- API schema (Jina / Cohere / vLLM compatible) -------------------------------------------------
class TextDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str = Field(min_length=1)
    documents: List[Union[str, TextDocument]] = Field(min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = True
    instruction: Optional[str] = Field(default=None, description="task instruction (Qwen3-Reranker extension)")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    margin: float
    truncated: Optional[bool] = None
    document: Optional[TextDocument] = None


class Usage(BaseModel):
    total_tokens: int


class RerankResponse(BaseModel):
    id: str
    object: str = "rerank"
    model: str
    results: List[RerankResult]
    usage: Usage
    meta: dict


class ApiError(Exception):
    def __init__(self, status, message, err_type="invalid_request_error", code=None):
        super().__init__(message)
        self.status, self.message, self.err_type, self.code = status, message, err_type, code


def error_body(message, err_type, code):
    return {"error": {"message": message, "type": err_type, "code": code}}


# ---- service ------------------------------------------------------------------------------------
class RerankService:
    def __init__(self, reranker, model_name, max_documents):
        self.rr = reranker
        self.model_name = model_name
        self.max_documents = max_documents
        self.npu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npu")  # one pair at a time
        self.created = int(time.time())

    async def rerank(self, req: RerankRequest) -> RerankResponse:
        if len(req.documents) > self.max_documents:
            raise ApiError(413, f"too many documents: {len(req.documents)} > {self.max_documents} (--max-documents)",
                           code="too_many_documents")
        texts = [d if isinstance(d, str) else d.text for d in req.documents]
        t0 = time.perf_counter()
        ranked = await asyncio.get_running_loop().run_in_executor(
            self.npu, lambda: self.rr.rerank(req.query, texts, instruction=req.instruction))
        elapsed = time.perf_counter() - t0
        results = [RerankResult(index=r.index, relevance_score=r.score, margin=r.margin,
                                truncated=True if r.truncated else None,
                                document=TextDocument(text=r.text) if req.return_documents else None)
                   for r in (ranked[: req.top_n] if req.top_n else ranked)]
        return RerankResponse(id=f"rerank-{uuid.uuid4().hex}", model=self.model_name, results=results,
                              usage=Usage(total_tokens=sum(r.num_tokens for r in ranked)),
                              meta={"elapsed_s": round(elapsed, 3), "documents": len(texts)})

    def close(self):
        self.npu.shutdown(wait=True)
        self.rr.close()


def create_app(service_factory, api_key=None):
    """service_factory() -> RerankService; called once at startup (loads the model onto the NPU)."""
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["service"] = service_factory()
        yield
        state.pop("service").close()

    app = FastAPI(title="Qwen3-Reranker (QNN)", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        return JSONResponse(error_body(e.message, e.err_type, e.code), status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, e: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in e.errors())
        return JSONResponse(error_body(msg or "invalid request", "invalid_request_error", None), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, e: StarletteHTTPException):  # 404 / 405 in OpenAI error format
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code)
        return JSONResponse(error_body(f"{e.detail}: {request.method} {request.url.path}", "invalid_request_error", code),
                            status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _server_error(request: Request, e: Exception):
        return JSONResponse(error_body(f"{type(e).__name__}: {e}", "server_error", None), status_code=500)

    def check_auth(request: Request):
        if api_key and request.headers.get("Authorization", "") != f"Bearer {api_key}":
            raise ApiError(401, "invalid or missing API key", "authentication_error", "invalid_api_key")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": state["service"].model_name}

    api = APIRouter(dependencies=[Depends(check_auth)])

    @api.get("/v1/models")
    async def models():
        s = state["service"]
        return {"object": "list", "data": [{"id": s.model_name, "object": "model", "created": s.created,
                                             "owned_by": "qwen3-reranker-qnn"}]}

    async def rerank(req: RerankRequest):
        return await state["service"].rerank(req)

    for path in ("/v1/rerank", "/rerank", "/v2/rerank"):
        api.add_api_route(path, rerank, methods=["POST"], response_model=RerankResponse, response_model_exclude_none=True)
    app.include_router(api)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("RERANKER_API_KEY"),
                    help="require 'Authorization: Bearer <key>' (default: env RERANKER_API_KEY, none = open)")
    ap.add_argument("--served-model-name", default="Qwen3-Reranker-0.6B")
    ap.add_argument("--variant", default=None, help="variant from variants.json (default: the first one)")
    ap.add_argument("--max-documents", type=int, default=64, help="documents per request (~4.6 s each at 4k)")
    ap.add_argument("--no-burst", action="store_true", help="do not vote for max HTP clocks")
    ap.add_argument("--log-level", type=int, default=1, help="QNN log level 0-3")
    args = ap.parse_args()

    import uvicorn
    from qwen3_reranker import Qwen3Reranker

    def factory():
        t0 = time.perf_counter()
        kwargs = {"variant": args.variant} if args.variant else {}  # default: the library's default variant
        rr = Qwen3Reranker(HERE, burst=not args.no_burst, log_level=args.log_level, **kwargs)
        print(f"[serve] loaded {rr.variant} ({rr.rt.num_parts} parts, seq_len {rr.seq_len}) in "
              f"{time.perf_counter() - t0:.1f}s; POST /v1/rerank, GET /v1/models, /health, /docs"
              f"{' (API key required)' if args.api_key else ''}", flush=True)
        return RerankService(rr, args.served_model_name, args.max_documents)

    # exactly one worker process: each process would load its own copy of the model onto the NPU
    uvicorn.run(create_app(factory, args.api_key), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
