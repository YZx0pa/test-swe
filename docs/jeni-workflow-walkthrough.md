# Jeni workflow: node by node, with the middleware at each step

Solid arrows always run. Dashed arrows are a choice the graph makes at run time.

---

### 1. `__start__`
A new recruiter message, or a resume after an approval, enters the graph.
- → `ModelCallLimitMiddleware.before_model` (solid)

---

### 2. `ModelCallLimitMiddleware.before_model`: step budget
**Middleware: ModelCallLimitMiddleware**
- **When:** before every model call.
- **How:** checks how many model calls this run has used. A run is one user message, or one resume after approval. The limit is 12.
- **Problem it solves:** an AI stuck in a loop running up cost forever.

**Arrows**
- ┄► `model`: the budget has room, so the AI can think.
- ┄► `TaskMemory.after_agent`: the budget is used up, so the run stops cleanly.

---

### 3. `model`: the AI decides
**Middleware inside this node: TaskMemoryMiddleware (wrap_model_call)**
- **When:** just before each model call.
- **How:** the full conversation stays in the checkpointer for the UI and audit. The model is given only **a short digest of earlier finished tasks plus the current task's messages**.
- **Problem it solves:** long threads cost too many tokens, old tasks get confused with new ones, and follow-ups like "now share them with me" still need to work.

**What the AI produces**
- **Tool calls**: it *proposes* an action. Nothing runs yet.
- **No tool call**: it replies directly with a final message:
  - `done`: the task is finished (including "not supported" or "not found");
  - `needs_user`: a question for the recruiter, for example "Which of these two jobs?"

**Arrow**
- → `ModelCallLimit.after_model` (solid), always.

---

### 4. `ModelCallLimitMiddleware.after_model`
**Middleware: ModelCallLimitMiddleware**
- **How:** records the call against the budget.

**Arrow**
- → `HumanInTheLoopMiddleware.after_model` (solid), always.

---

### 5. `HumanInTheLoopMiddleware.after_model`: approval gate
**Middleware: HumanInTheLoopMiddleware (Human-in-the-Loop, "HITL")**
- **When:** runs after **every** model call. The node always runs, but it only shows a card when a proposed tool needs approval. For any other tool it just passes the call through.

| Tool | Default (`gate_writes=False`) | `--approve-all` (CLI) / `JENI_GATE_WRITES=true` (demo) |
|---|---|---|
| Shortlist, reject, share application, transfer job ownership (`ALWAYS_CONFIRM`) | **card, every time** | **card, every time** |
| Other writes (`create_job`, `add_job_skills`, …) | no card | card |
| Read-only lookups | no card | no card |
- **How:** the graph **pauses** and the recruiter sees a plain-English card, for example "Shortlist applications 11, 12 and 13."
  - **Approve**: the call runs as proposed.
  - **Edit**: the call runs with the reviewer's arguments.
  - **Reject**: the call is cancelled, and a "rejected" note becomes its result.
  - An edit or rejection **stands** until the recruiter writes again. The always-confirm actions get a new card every time.
- **Problem it solves:** irreversible, high-impact changes happening without a person deciding.
- What it decided here (no card, approved, or edited) determines **which entity check applies later in `tools`**.

**Arrow**
- → `TaskMemory.after_model` (solid), always.

---

### 6. `TaskMemoryMiddleware.after_model`: routing point
**Middleware: TaskMemoryMiddleware (after_model)**
- **How:** only writes a progress line to the stage log. **It doesn't decide the route.**
- **Why the routing arrows start here:** the framework attaches its routing to the **last** `after_model` hook. This hook is last because TaskMemory is first in the middleware list. So routing always happens **after** the budget and approval steps, and nothing can reach `tools` without passing the approval gate.

**Arrows (the framework's routing, based on the model's last message)**
- ┄► `tools`: there are tool calls to run.
- ┄► `ModelCallLimit.before_model`: nothing is left to run (for example, the reviewer rejected every call), so the AI rethinks.
- ┄► `TaskMemory.after_agent`: no tool call, so this is the final answer or a question for the recruiter.

---

### 7. `tools`: layered checks, then the tool runs
Before any tool runs, the call passes through **check layers we added as middleware in `build_agent()`**. Studio doesn't draw them as boxes because they wrap the tool call inside this node. They run in this order:

```
tools
 └─ ① Entity DB check → ② ToolCallGuard → ③ the tool, inside a PII boundary
```

**① Entity DB check: EntitiesDBValidationMiddleware (`entities_db_validation.py`)**
- **Applies to:** every **write**. Read-only lookups skip it.
- **How:** checks in Postgres, **for this company only**, that the job, applications, users and emails exist and are eligible, and that a collaborator role is 1 or 5.
- **Problem it solves:** acting on records that don't exist or belong to another company.
- **Always on the version that will run.** It's added as two layers around the approval step, so the outcome at node 5 decides which one checks the call:

| Outcome at node 5 | Checked by |
|---|---|
| No card (routine write) | the execution layer (EntityExec) |
| Card, approved unchanged | the pre-approval layer (EntityPre) |
| Card, reviewer edited | both: EntityPre on the original, EntityExec on the **edited** version |
| Card, rejected | not checked, because the tool never runs |

  HITL also wraps the tool here: when the reviewer edited the call, it swaps in their version, and that's what gets checked and run.

**② ToolCallGuard**
- **Applies to:** **every** tool call, including lookups, right before it runs.
- **Ids must come from somewhere real.** An id is accepted if it appears in:
  - what the recruiter typed in this task, or the reviewer's edit;
  - **any earlier tool result** in this task, for example a `job_id` returned by `find_job_by_title`;
  - the memory digest of earlier finished tasks.

  It refuses **invented** ids, and **wrong-kind** ids such as a profile id passed as an application id.
- **Emails and candidate details are stricter.** `emails`, `new_owner_user_email`, `candidate_email` and `candidate_name` must be what the **recruiter typed**. The only exceptions apply to share recipients and new owners:
  - a **colleague's email token** returned by the user directory (`search_users`, `find_user`);
  - **"me"**, the signed-in user.

  A candidate's email taken from a tool result is always refused.
- **Also refuses** exact repeat calls and redoing what a reviewer rejected or edited. A tool crash becomes an error result instead of a failed run.
- **Problem it solves:** made-up ids, guessed personal data, double-sending, and overriding a reviewer.

**③ The tool runs, inside a PII boundary (tool layer, not middleware)**
- **On the way out (decrypt at send):** the AI only ever handles email **tokens** (`<email:…>`) or `"me"`. `pii_vault` swaps them back to real addresses **just before the request is sent** to VIRA or the database. Real addresses stay encrypted in process memory and never enter the conversation, the checkpointer, traces or logs.
- **The call itself:**
  - **Database lookups** (`find_job_by_title`, `list_job_applications`): company-scoped SQL. `company_id` is set by the system, never by the AI.
  - **VIRA lookups and actions**: schema check → one task sent to VIRA through the guarded path → the **real outcome** is read back, so "submitted" is never reported as "done".
- **On the way back (mask):** every result is masked before the AI sees it (`recruiter_cli._mask_pii`).
  - Emails become tokens. A colleague's from the user directory can be used as a recipient; a candidate's can't.
  - Names, phone numbers, resumes, addresses and other personal fields are redacted. Emails and phone numbers inside free text are scrubbed too.
- **Audit log:** each call is recorded with personal data redacted, in an owner-only file.
- **Problem it solves:** the AI can act on people (share with a colleague, transfer to "me") without ever seeing personal data. Masking sits in the tool layer, so it holds in every runtime, not only LangGraph.

If any layer refuses the call, the tool **doesn't run**, and the AI gets a clear reason. If the database can't be reached, the action also doesn't run.

**Arrows**
- ┄► `ModelCallLimit.before_model`: the normal case. The result, or the refusal reason, goes back to the AI, which reads it and decides the next step. **This is the main loop.**
- ┄► `TaskMemory.after_agent`: drawn by the framework for a tool that ends the run directly. Rarely used here.

---

### 8. `TaskMemoryMiddleware.after_agent`: memory
**Middleware: TaskMemoryMiddleware (after_agent)**
- **When:** once, when the run finishes.
- **How:**
  - `done`: saves a compact record of the task (instruction, lookups, actions, outcome) and starts a new task boundary.
  - `needs_user`: keeps the task **open**, so the recruiter's answer continues the same task.
- **Problem it solves:** the next request can build on earlier ones without re-sending the whole history, and a question and its answer stay one task.

**Arrow**
- → `__end__` (solid)

---

### 9. `__end__`
The recruiter gets the reply.

---

### One-line close
> "The AI only proposes the next step. Around it, middleware controls cost (step budget), risk (human approval), data correctness (database checks and the guard: every id from a real source, personal data only from the recruiter), privacy (email tokens and masking) and context (task memory). Nothing can skip it."
