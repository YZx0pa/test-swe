
from __future__ import annotations

import argparse
import asyncio
import hmac
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator, model_validator

import agent_kit
import run_langgraph

_THREAD_ID = r"^[A-Za-z0-9_-]{8,64}$"
_CONTEXT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_CONTEXT_MAX_KEYS, _CONTEXT_MAX_CHARS = 10, 200

# X-Client-Name -> the environment variable holding that client's Postgres DSN.
ClientName = Literal["tron", "live"]
CLIENT_DSN_VARS: dict[str, str] = {"tron": "TRON_POSTGRES_DSN", "live": "LIVE_POSTGRES_DSN"}
# The API key every caller sends, whatever the client.
API_KEY_VAR = "JENI_API_KEY"


# def make_checkpointer():
#     """The threads store shared by every company's agent.  In memory for now; a Postgres
#     saver (langgraph-checkpoint-postgres) can replace it here without touching the API."""
#     return InMemorySaver()


# --- request / response bodies ----------------------------------------------
class EditedAction(BaseModel):
    name: str
    args: dict[str, Any]


class Decision(BaseModel):
    """One answer to one action request on an approval card (HumanInTheLoopMiddleware's)."""
    type: Literal["approve", "edit", "reject"]
    edited_action: EditedAction | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _edit_has_action(self):
        if self.type == "edit" and self.edited_action is None:
            raise ValueError("an edit decision needs edited_action: {name, args}")
        return self


class ChatRequest(BaseModel):
    """One turn: what the user typed, or -- while an approval waits -- explicit decisions."""
    user_input: str | None = Field(default=None, min_length=1)
    thread_id: str | None = Field(default=None, pattern=_THREAD_ID)
    context: dict[str, int | str] = Field(default_factory=dict)
    # Answers to the waiting approval card: one per action request (approve / edit / reject).
    # Several waiting at once: {interrupt_id: [decision, ...]}.
    decisions: list[Decision] | dict[str, list[Decision]] | None = None

    @model_validator(mode="after")
    def _input_or_decisions(self):
        if (self.user_input is None) == (self.decisions is None):
            raise ValueError("send either user_input or decisions, not both")
        if self.decisions is not None and self.thread_id is None:
            raise ValueError("decisions need the thread_id of the approval they answer")
        return self

    @field_validator("thread_id", mode="before")
    @classmethod
    def _blank_is_new(cls, value):
        # A UI's empty field means "no thread yet": start one, as if thread_id were left out.
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("context")
    @classmethod
    def _small_and_plain(cls, value: dict[str, int | str]) -> dict[str, int | str]:
        # It goes into the model's input as text: keep it to short, plain key/value pairs.
        if len(value) > _CONTEXT_MAX_KEYS:
            raise ValueError(f"context takes at most {_CONTEXT_MAX_KEYS} keys")
        for key, item in value.items():
            if not _CONTEXT_KEY.match(key):
                raise ValueError(f"context key {key!r} must be snake_case, up to 40 characters")
            if isinstance(item, str) and (len(item) > _CONTEXT_MAX_CHARS
                                          or any(c in item for c in "\n\r[]")):
                raise ValueError(f"context value of {key!r} must be one short line, no brackets")
        return value


class Approval(BaseModel):
    interrupt_id: str
    action_requests: list[dict[str, Any]]
    review_configs: list[dict[str, Any]] = []


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["done", "needs_user", "approval_required"]
    reply: str
    approvals: list[Approval] = []


# --- helpers ------------------------------------------------------------------
def with_page_context(user_input: str, context: dict[str, int | str]) -> str:
    """The turn's message as the model sees it: the UI page first, then what the user typed."""
    if not context:
        return user_input
    page = ", ".join(f"{key}={value}" for key, value in context.items())
    return f"[page context: {page}]\n{user_input}"


def reply_from(result: dict) -> tuple[str, str]:
    """(status, reply) from a finished run: the TerminalResponse, else the last AI text."""
    terminal = agent_kit.terminal_response_data(result)
    if terminal:
        return terminal["status"], terminal["message"]
    return "done", agent_kit.final_text(result)


def approvals_from(interrupts) -> list[Approval]:
    return [Approval(interrupt_id=i.id, action_requests=i.value.get("action_requests", []),
                     review_configs=i.value.get("review_configs", []))
            for i in interrupts]


APPROVAL_REPLY = "Jeni needs approval before it continues."


# A typed answer to a waiting approval, as in the terminal (agent_kit.ask_human): a fixed yes
# or no, never free text.  Edits go through decisions with edited_action.
_YES, _NO = frozenset({"y", "yes"}), frozenset({"n", "no"})


def answer_kind(user_input: str) -> Literal["yes", "no", "other"]:
    """A typed reply to a waiting approval: yes / no, or anything else (asked again)."""
    word = user_input.strip().lower().rstrip(".!")
    return "yes" if word in _YES else "no" if word in _NO else "other"


# --- the service --------------------------------------------------------------
@dataclass
class _Thread:
    client: str
    company_id: int
    user_id: int
    pending: dict[str, int] = field(default_factory=dict)   # interrupt id -> action requests
    approvals: list[Approval] = field(default_factory=list)   # the open cards, while pending
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class JeniService:
    """Agents per company, threads per conversation, and one run per request."""

    def __init__(self):
        # The chat model is build_agent's default, as in run_langgraph: $CHAT_MODEL.
        # self.checkpointer = checkpointer if checkpointer is not None else make_checkpointer()
        self.pools: dict[str, Any] = {}             # client name -> asyncpg pool
        self.api_key: str | None = None             # what callers send as the Bearer token
        self._agents: dict[tuple[str, int], Any] = {}
        self._threads: dict[str, _Thread] = {}
        self._building = asyncio.Lock()

    async def start(self) -> None:

        agent_kit.setup(argparse.Namespace( mode="real",
                                           stage_log="logs/jeni_agent.log"))    
        needed = [*CLIENT_DSN_VARS.values(), API_KEY_VAR]
        missing = [var for var in needed if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"the API needs {', '.join(missing)} to be set")
        self.api_key = os.environ[API_KEY_VAR]
        try:
            for client, var in CLIENT_DSN_VARS.items():   # on this server's event loop
                self.pools[client] = await run_langgraph._open_pool(os.environ[var])
                agent_kit.stage("API_POOL_OPENED", client=client)
        except BaseException:                 # _open_pool raises SystemExit: close what opened
            await self.stop()
            raise

    async def stop(self) -> None:
        pools, self.pools = self.pools, {}
        for pool in pools.values():
            await pool.close()

    async def agent_for(self, client: str, company_id: int):
        async with self._building:
            key = (client, company_id)
            if key not in self._agents:
                self._agents[key] = self._build(client, company_id)
            return self._agents[key]


    def _query_tools(self, client: str):
        """The client's db queries: entity validation's lookups (not given to the model)."""
        from db_queries import build_db_queries
        return build_db_queries(self.pools[client])

    def _build(self, client: str, company_id: int):
        context = {"auth_profile": {"company_id": company_id}}
        query_tools = self._query_tools(client)
        # toolset= agent_kit.toolset("jeni_db",query_tools,context)
        toolset= agent_kit.toolset("jeni")
        return  run_langgraph.build_agent(approve_all=True, 
                                        step_limit=12,
                                        toolset=toolset,
                                        query_tools=query_tools, 
                                        context=context,
                                        task_memory=True,
                                        checkpointer=InMemorySaver())

    def _thread(self, thread_id: str | None, client: str, company_id: int, user_id: int,
                *, claim: bool) -> tuple[str, _Thread]:
        """The caller's thread: a new id when none is given; an unused id is claimed."""
        if thread_id is None:
            thread_id = uuid.uuid4().hex
        thread = self._threads.get(thread_id)
        if thread is None:
            if not claim:
                raise HTTPException(404, "unknown thread_id")
            thread = self._threads.setdefault(thread_id, _Thread(client, company_id, user_id))
        if (thread.client, thread.company_id, thread.user_id) != (client, company_id, user_id):
            raise HTTPException(403, "this thread belongs to another client, company or user")
        return thread_id, thread

    async def chat(self, client: str, company_id: int, user_id: int,
                   body: ChatRequest) -> ChatResponse:
        if body.decisions is not None:          # buttons: approve / edit / reject
            return await self._decide(client, company_id, user_id, body)
        thread_id, thread = self._thread(body.thread_id, client, company_id, user_id, claim=True)
        async with thread.lock:
            agent = await self.agent_for(client, company_id)
            if thread.pending:                  # a typed yes / no answers the waiting approval
                kind = answer_kind(body.user_input)
                agent_kit.stage("API_APPROVAL", thread=thread_id, answer=kind)
                if kind == "other":             # not a yes or no: nothing changes, same card again
                    return ChatResponse(thread_id=thread_id, status="approval_required",
                                        reply=f"{APPROVAL_REPLY} Please answer yes or no.",
                                        approvals=thread.approvals)
                decision = ({"type": "approve"} if kind == "yes" else
                            {"type": "reject", "message": "The user declined this call."})
                return await self._run(agent, thread_id, thread,
                                       Command(resume=self._resume_all(thread, decision)))
            text = with_page_context(body.user_input, body.context)
            return await self._run(agent, thread_id, thread,
                                   {"messages": [{"role": "user", "content": text}]},
                                   context_keys=",".join(sorted(body.context)) or None)

    async def _decide(self, client: str, company_id: int, user_id: int,
                      body: ChatRequest) -> ChatResponse:
        """Explicit decisions for the waiting approval card: approve, edit or reject."""
        thread_id, thread = self._thread(body.thread_id, client, company_id, user_id,
                                         claim=False)
        async with thread.lock:
            if not thread.pending:
                raise HTTPException(409, "no approval is pending on this thread")
            given = body.decisions
            if isinstance(given, list):
                if len(thread.pending) != 1:
                    raise HTTPException(400, "several approvals are pending: send decisions as "
                                             "{interrupt_id: [decision, ...]}")
                given = {next(iter(thread.pending)): given}
            if set(given) != set(thread.pending):
                raise HTTPException(400, f"decisions must answer exactly these approvals: "
                                         f"{sorted(thread.pending)}")
            for interrupt_id, decisions in given.items():
                if len(decisions) != thread.pending[interrupt_id]:
                    raise HTTPException(400, f"approval {interrupt_id} has "
                                             f"{thread.pending[interrupt_id]} action request(s): "
                                             f"send one decision for each")
            answers = {iid: {"decisions": [d.model_dump(exclude_none=True) for d in ds]}
                       for iid, ds in given.items()}
            resume = next(iter(answers.values())) if len(answers) == 1 else answers
            agent = await self.agent_for(client, company_id)
            return await self._run(agent, thread_id, thread, Command(resume=resume))

    @staticmethod
    def _resume_all(thread: _Thread, decision: dict):
        """The same decision for every action request on every open card."""
        answers = {interrupt_id: {"decisions": [dict(decision) for _ in range(count)]}
                   for interrupt_id, count in thread.pending.items()}
        return next(iter(answers.values())) if len(answers) == 1 else answers

    async def _run(self, agent, thread_id: str, thread: _Thread, payload,
                   context_keys: str | None = None) -> ChatResponse:
        """One run of the graph, up to its reply or its next approval card (never prompts)."""
        counter = agent_kit.UsageCounter()
        config = {"configurable": {"thread_id": thread_id},
                  "callbacks": [counter]}
        agent_kit.stage("API_REQUEST", thread=thread_id, client=thread.client,
                        company_id=thread.company_id,
                        user_id=thread.user_id, context=context_keys,
                        kind="resume" if isinstance(payload, Command) else "message")
        try:
            out = await agent.ainvoke(payload, config, version="v2")
        except Exception as exc:              # the message may name hosts or carry data
            agent_kit.stage("API_RESPONSE", thread=thread_id, status="error",
                            error=type(exc).__name__)
            raise HTTPException(502, f"the agent run failed ({type(exc).__name__})") from None
        if out.interrupts:
            thread.pending = {i.id: len(i.value.get("action_requests", [])) for i in out.interrupts}
            thread.approvals = approvals_from(out.interrupts)
            response = ChatResponse(thread_id=thread_id, status="approval_required",
                                    reply=APPROVAL_REPLY, approvals=thread.approvals)
        else:
            thread.pending, thread.approvals = {}, []
            status, reply = reply_from(out.value)
            response = ChatResponse(thread_id=thread_id, status=status, reply=reply)
        agent_kit.stage("API_RESPONSE", thread=thread_id, status=response.status,
                        model_calls=counter.calls, input_tokens=counter.input_tokens,
                        output_tokens=counter.output_tokens)
        return response


# --- the app ------------------------------------------------------------------
def client_name(request: Request,
                x_client_name: ClientName = Header(..., alias="X-Client-Name"),
                authorization: str | None = Header(None)) -> str:
    """The X-Client-Name header, once the caller shows the API key: which client's pool the
    request's agent queries."""
    expected = request.app.state.jeni.api_key
    scheme, _, key = (authorization or "").partition(" ")
    if not (expected and scheme.lower() == "bearer"
            and hmac.compare_digest(key.strip().encode(), expected.encode())):
        raise HTTPException(401, "a valid API key is required",
                            headers={"WWW-Authenticate": "Bearer"})
    return x_client_name


def create_app(service: JeniService | None = None) -> FastAPI:
    service = service or JeniService()


    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="Jeni", lifespan=lifespan)
    app.state.jeni = service

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest,
                   company_id: int = Query(..., gt=0),
                   user_id: int = Query(..., gt=0),                   
                   client: str = Depends(client_name)) -> ChatResponse:
        return await service.chat(client, company_id, user_id, body)

    return app


app = create_app()
