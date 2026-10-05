"""The demo's script: what the presenter types, what they do at each approval card, and what
the data must show after each act.

demo/SCRIPT.md is the presenter's copy (a test keeps its prompts the same as these),
demo/rehearse.py plays it against the running server and checks the data, and the chat UI
offers each act's first prompt as a starter (GET /demo/prompts), grouped by category.  Acts 1-8
are the presented run; 9-17 are "more to try", for questions or a longer session.

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


# The chat UI's starter groups, in this order.
CATEGORIES = ("Jobs", "Applicants", "Team & sharing", "Guardrails")


@dataclass(frozen=True)
class Act:
    key: str
    title: str
    turns: tuple[Turn, ...]
    check: Check
    expect: str                                          # what the check wants, in words
    category: str
    # At an approval card, per tool: {"type": "reject", "message": ...} or
    # {"type": "edit", "args": {...}} (merged into the proposed args).  Anything else: approve.
    decide: dict = field(default_factory=dict)
    # Approval cards the act must show, for acts whose data ends unchanged or edited.
    cards: tuple[str, ...] = ()


def writes(activity: list, *, completed: bool = True) -> list[str]:
    return [e["task"] for e in activity
            if e["kind"] == "write" and (e["status"] == "completed" or not completed)]


def reads(activity: list) -> list[tuple[str, dict]]:
    return [(e["task"], e["fields"]) for e in activity
            if e["kind"] == "read" and e["status"] == "completed"]


def _lower(values) -> set[str]:
    return {str(v).strip().lower() for v in values or []}


def _lookup(jobs, activity):
    return ({"kubernetes", "terraform"} <= _lower(jobs[7001]["skills"])
            and writes(activity) == ["add_job_skills"])


def _reject(jobs, activity):
    return jobs[7003]["owner"] is None and writes(activity, completed=False) == []


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


def _clone(jobs, activity):
    cloned = [e for e in activity if e["task"] == "clone_job" and e["status"] == "completed"]
    if writes(activity) != ["clone_job", "add_job_skills"] or len(cloned) != 1:
        return False
    copy = jobs.get(cloned[0]["created"])
    added = [e for e in activity if e["task"] == "add_job_skills"][0]
    return (copy is not None and "rust" in _lower(copy["skills"])
            and "rust" not in _lower(jobs[7001]["skills"])
            and added["fields"].get("job_id") == copy["jobId"])


def _tidy(jobs, activity):
    return (sorted(writes(activity)) == ["make_job_public", "remove_job_skills"]
            and "postgresql" not in _lower(jobs[7001]["skills"])
            and {"python", "go"} <= _lower(jobs[7001]["skills"]) and jobs[7001]["isPrivate"] is False)


def _edit(jobs, activity):
    edits = [e["fields"] for e in activity if e["task"] == "edit_job" and e["status"] == "completed"]
    return (writes(activity) == ["edit_job"] and edits[0].get("job_id") == 7003
            and (edits[0].get("vacancy"), edits[0].get("min_exp"), edits[0].get("max_exp")) == (2, 2, 4))


def _close(jobs, activity):
    closed = [e["fields"] for e in activity if e["task"] == "make_job_closed" and e["status"] == "completed"]
    return (writes(activity) == ["make_job_closed"] and jobs[7003]["status"] == "closed"
            and bool(closed[0].get("reason_for_closure")))


def _reject_weakest(jobs, activity):
    stages = {a["appId"]: a["stage"] for a in jobs[7001]["applications"]}
    return (writes(activity) == ["reject_multiple_application"] and stages[5104] == "rejected"
            and all(stages[a] == "applied" for a in (5101, 5102, 5103)))


def _compare(jobs, activity):
    return writes(activity, completed=False) == [] and any(
        task == "get_applications" and fields.get("job_id") == 7002 for task, fields in reads(activity))


def _suggest(jobs, activity):
    return writes(activity, completed=False) == [] and any(
        task == "get_suggested_candidates_for_a_job" and fields.get("job_id") == 7001
        for task, fields in reads(activity))


def _me(jobs, activity):
    shared = [e for e in activity if e["task"] == "share_application" and e["status"] == "completed"]
    return (writes(activity) == ["share_application"]
            and shared[0]["fields"] == {"app_ids": [5103], "recipients": 1})


def _hand_over(jobs, activity):
    return writes(activity) == ["transfer_job_ownership"] and jobs[7002]["owner"] == "Priya Nair"


ACTS = (
    Act("lookup", "Finds the job by name and makes a routine change",
        (Turn("Add Kubernetes and Terraform to the backend engineer job."),),
        _lookup, "job 7001 gains Kubernetes and Terraform; one write", "Jobs"),
    Act("reject", "High-stakes changes wait for a person, who can say no",
        (Turn("Transfer ownership of the Product Designer job to alice.johnson@example.com."),),
        _reject, "the transfer is rejected at its card: nothing reaches VIRA, job 7003 keeps its owner",
        "Guardrails",
        decide={"transfer_job_ownership": {"type": "reject",
                                           "message": "Not yet: Alice starts next month."}},
        cards=("transfer_job_ownership",)),
    Act("chain", "Uses what one step returns in the next",
        (Turn("Create a Data Engineer job in Singapore needing Python and SQL, with 3 to 5 years "
              "of experience, then add Spark to it."),),
        _chain, "create_job, then add_job_skills on the id create_job returned", "Jobs"),
    Act("shortlist", "Reads the data to decide, and the reviewer can change the call",
        (Turn("Shortlist the two strongest applicants for the backend engineer job."),),
        _shortlist, "the reviewer edits the shortlist to 5102 only: 5102 shortlisted, 5103 not",
        "Applicants",
        decide={"shortlist_multiple_application": {"type": "edit", "args": {"app_ids": [5102]}}},
        cards=("shortlist_multiple_application",)),
    Act("ask", "Asks only for what it can't look up, then carries on",
        (Turn("Add Bob to the hiring team for the backend engineer job."),
         Turn("As a team member.", if_needed=True)),
        _ask, "Bob (802) joins job 7001 as a team member, not an administrator; no transfer",
        "Team & sharing"),
    Act("email", "Never guesses personal data",
        (Turn("Share application 5102 with the hiring manager."),
         Turn("Send it to priya.nair@example.com with a note: strong backend profile, worth a call.",
              if_needed=True)),
        _email, "one share of application 5102, to the one address the presenter typed", "Guardrails"),
    Act("recover", "Knows LinkedIn needs an open job, and does both steps when told",
        (Turn("Publish the Data Analyst job to LinkedIn."),
         Turn("Reopen it, then publish it.", if_needed=True)),
        _recover, "job 7002 ends open and on LinkedIn: reopened, then published", "Jobs"),
    Act("unsupported", "Says when something isn't supported",
        (Turn("Share the Product Designer job on Facebook."),),
        _unsupported, "nothing changes: sharing a job isn't a Jeni task", "Guardrails"),
    # More to try (SCRIPT.md): not in the presented run.
    Act("clone", "Copies a job and changes only the copy",
        (Turn("Copy the backend engineer job and add Rust to the copy only."),),
        _clone, "clone_job, then add_job_skills on the copy's id: the copy has Rust, job 7001 doesn't",
        "Jobs"),
    Act("tidy", "Makes two routine changes from one request",
        (Turn("Remove PostgreSQL from the backend engineer job and make it public."),),
        _tidy, "job 7001 loses PostgreSQL, keeps Python and Go, and turns public", "Jobs"),
    Act("edit", "Edits a job's details",
        (Turn("Set the Product Designer job to 2 openings, needing 2 to 4 years of experience."),),
        _edit, "one edit_job on job 7003: 2 openings, 2 to 4 years", "Jobs"),
    Act("close", "Closes a job and records why",
        (Turn("Close the Product Designer job: the role has been filled."),),
        _close, "job 7003 closed, with a reason", "Jobs"),
    Act("reject_weakest", "Rejects an applicant once a person approves",
        (Turn("Reject the weakest applicant for the backend engineer job."),),
        _reject_weakest, "the card is approved: 5104 (0.64) rejected, the other three still applied",
        "Applicants", cards=("reject_multiple_application",)),
    Act("compare", "Answers from the data without changing anything",
        (Turn("Which applicant for the Data Analyst job has the best match score, and what stage "
              "are they at?"),),
        _compare, "job 7002's applications read; nothing changes (the answer: 5201, shortlisted)",
        "Applicants"),
    Act("suggest", "Finds suggested candidates for a job",
        (Turn("Find suggested candidates for the backend engineer job."),),
        _suggest, "job 7001's suggested candidates read; nothing changes", "Applicants"),
    Act("me", "Knows who \"me\" is",
        (Turn("Share application 5103 with me, with a note: follow up next week."),),
        _me, "the card says \"with you\" and is approved: one share of 5103 to one recipient",
        "Team & sharing", cards=("share_application",)),
    Act("hand_over", "Looks a colleague up and hands a job over, once approved",
        (Turn("Transfer ownership of the Data Analyst job to Priya."),),
        _hand_over, "the card is approved: Priya Nair owns job 7002", "Team & sharing",
        cards=("transfer_job_ownership",)),
)


def prompts() -> list[dict]:
    """Each act's first prompt, for the chat UI's starters: numbered as in SCRIPT.md, grouped
    by category."""
    numbered = sorted(enumerate(ACTS, 1), key=lambda na: (CATEGORIES.index(na[1].category), na[0]))
    return [{"key": a.key, "number": n, "category": a.category, "title": a.title,
             "prompt": a.turns[0].say} for n, a in numbered]


def decision(act: Act, action: dict) -> dict:
    """The decision for one proposed call at an approval card, in HumanInTheLoopMiddleware's
    format."""
    planned = act.decide.get(action["name"], {"type": "approve"})
    if planned["type"] == "edit":
        return {"type": "edit", "edited_action": {"name": action["name"],
                                                  "args": {**action["args"], **planned["args"]}}}
    return dict(planned)
