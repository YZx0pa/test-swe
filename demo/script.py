"""The demo's script: what the presenter types, what they do at each approval card, and what
the data must show after each act.

demo/SCRIPT.md is the presenter's copy (a test keeps its prompts the same as these),
demo/rehearse.py plays it against the running server and checks the data, and the chat UI
offers each act's first prompt as a starter (GET /demo/prompts).

Each act runs in a new conversation, in order, from the starting data.  A check reads the
data panel's snapshot (jobs by id) and the sub-tasks that reached the mock during the act:
what really changed, not what the model said.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Check = Callable[[dict, list], bool]


@dataclass(frozen=True)
class Turn:
    say: str
    # Only when the act's check doesn't pass yet: the answer to a question the agent may ask.
    if_needed: bool = False


@dataclass(frozen=True)
class Act:
    key: str
    title: str
    turns: tuple[Turn, ...]
    check: Check
    expect: str                                          # what the check wants, in words
    # At an approval card, per tool: {"type": "reject", "message": ...} or
    # {"type": "edit", "args": {...}} (merged into the proposed args).  Anything else: approve.
    decide: dict = field(default_factory=dict)
    # Approval cards the act must show, for acts whose data ends unchanged or edited.
    cards: tuple[str, ...] = ()


def writes(activity: list, *, completed: bool = True) -> list[str]:
    return [e["task"] for e in activity
            if e["kind"] == "write" and (e["status"] == "completed" or not completed)]


def _lower(values) -> set[str]:
    return {str(v).strip().lower() for v in values or []}


def _lookup(jobs, activity):
    return ({"kubernetes", "terraform"} <= _lower(jobs[7001]["skills"])
            and writes(activity) == ["add_job_skills"])


def _reject(jobs, activity):
    return jobs[7003]["status"] == "open" and writes(activity, completed=False) == []


def _chain(jobs, activity):
    created = [e for e in activity if e["task"] == "create_job" and e["status"] == "completed"]
    if writes(activity) != ["create_job", "add_job_skills"] or len(created) != 1:
        return False
    job = jobs.get(created[0]["created"])
    added = [e for e in activity if e["task"] == "add_job_skills"][0]
    return (job is not None and "data engineer" in job["jobName"].lower()
            and {"python", "sql", "spark"} <= _lower(job["skills"])
            and added["fields"].get("job_id") == job["jobId"])


def _shortlist(jobs, activity):
    stages = {a["appId"]: a["stage"] for a in jobs[7001]["applications"]}
    return (writes(activity) == ["shortlist_multiple_application"]
            and stages[5102] == "shortlisted" and stages[5103] == "applied")


def _ask(jobs, activity):
    return (writes(activity) == ["add_job_collaborators"]
            and [(t["userId"], t["role"]) for t in jobs[7001]["team"]] == [(802, "team member")])


def _email(jobs, activity):
    shared = [e for e in activity if e["task"] == "share_application" and e["status"] == "completed"]
    return (writes(activity) == ["share_application"]
            and shared[0]["fields"] == {"app_ids": [5102], "recipients": 1})


def _recover(jobs, activity):
    done = writes(activity)
    return (jobs[7002]["status"] == "open" and jobs[7002]["linkedIn"]
            and done[-2:] == ["make_job_open", "publish_job_to_linkedin"])


def _unsupported(jobs, activity):
    return writes(activity, completed=False) == []


ACTS = (
    Act("lookup", "Finds the job by name, and asks before it changes anything",
        (Turn("Add Kubernetes and Terraform to the backend engineer job."),),
        _lookup, "job 7001 gains Kubernetes and Terraform; one write"),
    Act("reject", "A person can say no",
        (Turn("Close the Product Designer job, we've filled it."),),
        _reject, "the close is rejected at the card: nothing reaches VIRA, job 7003 stays open",
        decide={"make_job_closed": {"type": "reject",
                                    "message": "Not yet: the hiring manager is still interviewing."}},
        cards=("make_job_closed",)),
    Act("chain", "Uses what one step returns in the next",
        (Turn("Create a Data Engineer job in Singapore needing Python and SQL, with 3 to 5 years "
              "of experience, then add Spark to it."),),
        _chain, "create_job, then add_job_skills on the id create_job returned"),
    Act("shortlist", "Reads the data to decide, and the reviewer can change the call",
        (Turn("Shortlist the two strongest applicants for the backend engineer job."),),
        _shortlist, "the reviewer edits the shortlist to 5102 only: 5102 shortlisted, 5103 not",
        decide={"shortlist_multiple_application": {"type": "edit", "args": {"app_ids": [5102]}}},
        cards=("shortlist_multiple_application",)),
    Act("ask", "Asks only for what it can't look up, then carries on",
        (Turn("Add Bob to the hiring team for the backend engineer job."),
         Turn("As a team member.", if_needed=True)),
        _ask, "Bob (802) joins job 7001 as a team member, not an administrator; no transfer"),
    Act("email", "Never guesses personal data",
        (Turn("Share application 5102 with the hiring manager."),
         Turn("Send it to priya.nair@example.com with a note: strong backend profile, worth a call.",
              if_needed=True)),
        _email, "one share of application 5102, to the one address the presenter typed"),
    Act("recover", "Knows LinkedIn needs an open job, and does both steps when told",
        (Turn("Publish the Data Analyst job to LinkedIn."),
         Turn("Reopen it, then publish it.", if_needed=True)),
        _recover, "job 7002 ends open and on LinkedIn: reopened, then published"),
    Act("unsupported", "Says when something isn't supported",
        (Turn("Share the Product Designer job on Facebook."),),
        _unsupported, "nothing changes: sharing a job isn't a Jeni task"),
)


def prompts() -> list[dict]:
    """Each act's first prompt, for the chat UI's starters."""
    return [{"key": a.key, "title": a.title, "prompt": a.turns[0].say} for a in ACTS]


def decision(act: Act, action: dict) -> dict:
    """The decision for one proposed call at an approval card, in HumanInTheLoopMiddleware's
    format."""
    planned = act.decide.get(action["name"], {"type": "approve"})
    if planned["type"] == "edit":
        return {"type": "edit", "edited_action": {"name": action["name"],
                                                  "args": {**action["args"], **planned["args"]}}}
    return dict(planned)
