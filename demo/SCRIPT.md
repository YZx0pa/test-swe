# Jeni v2 demo script

About 12 minutes: a 1-minute opening, eight short acts, a 2-minute close. Nine more acts
("More to try", 9–17) show the rest of what Jeni can do, for questions or a longer session.
Everything runs on mock VIRA with synthetic jobs, people and applicants. Nothing reaches VIRA or
a real database.

The prompts below are the ones `demo/rehearse.py` plays and checks, and the chat's starter cards
offer each act's first prompt, grouped as Jobs, Applicants, Team & sharing and Guardrails. Type
them as written, or click the card and press Enter. Start each act in a **new chat** (the pencil
icon, top right).

## Before the audience arrives

1. `demo/run.sh`, and wait for "Jeni v2 demo is up".
2. Rehearse once: `.venv/bin/python -m demo.rehearse` (all 17 acts: about 4 minutes, about
   $0.09). It should end `17/17`. For the eight presented acts only, add `--act` for each
   (`--act lookup --act reject …`). Every act starts from the starting data, and at the end it
   deletes its chats and resets the data.
3. Open <http://localhost:3000>. Zoom the browser to 110–125% so the back row can read it. The
   data panel is on the right; **Show data** / **Hide data** toggles it.
4. The chat never shows tool calls, their payloads or raw results. An approval card says in
   plain words what Jeni wants to do ("Shortlist applications 5102 and 5103."), and the data
   panel's "What reached VIRA" shows every call that ran.
5. If anything looks off, press **Reset data** in the panel. Every act works from the starting
   data.
6. Approvals: routine changes run straight away and the high-stakes ones (shortlist, reject,
   share a CV, transfer ownership) show a card. To show a card for every change, as on real
   VIRA, set `GATE_WRITES = True` in `demo/jeni_graph.py` and restart `demo/run.sh`; then
   approve each extra card.

Starting data, for your reference:

| Job | Id | State | Applicants (match score) |
|---|---|---|---|
| Senior Backend Engineer | 7001 | open, private | 5102 (0.91), 5103 (0.85), 5101 (0.72), 5104 (0.64) |
| Data Analyst | 7002 | closed, public | 5201 (0.88, shortlisted), 5202 (0.59) |
| Product Designer | 7003 | open, public | none |

Colleagues: Alice Johnson (801), Bob Tan (802), Priya Nair (803). You are Sam Lee (804):
"share it with me" or "transfer it to me" means Sam.

## Opening (1 minute)

- Jeni v1 plans a whole request in one model call, up front, and hands the plan to VIRA. It
  can't use what one step returns in the next, and it fills in defaults nobody chose.
- v2 works one step at a time and reads real results before deciding the next step. Routine
  changes run straight away; high-stakes ones (shortlisting, rejecting, sharing a CV,
  transferring ownership) always wait for a person. On real VIRA, every change does.
- Left: the chat. Right: the data. Watch the right side; it shows what really changed.

## 1. Finds the job by name and makes a routine change

Type: `Add Kubernetes and Terraform to the backend engineer job.`

- Jeni looks up "backend engineer", finds job 7001, and adds the skills straight away.

Point out:
- Nobody typed a job id.
- Adding skills is routine, so there's no card here. On real VIRA every change waits for one.
- In the panel, Senior Backend Engineer's skills gain Kubernetes and Terraform, and "What reached
  VIRA" shows the one write.

## 2. High-stakes changes wait for a person, who can say no

Type: `Transfer ownership of the Product Designer job to alice.johnson@example.com.`

- Jeni finds the job, and may check that Alice is a colleague. Then a card asks: "Transfer
  ownership of job 7003 to alice.johnson@example.com." Under **Reject**, type the reason
  `Not yet: Alice starts next month.` and click **Submit rejection**.

Point out:
- Transferring ownership, shortlisting, rejecting and sharing a CV always wait for a person, in
  every mode.
- Nothing reached VIRA: the feed has no new write, and the job has no new owner.
- Jeni reports what happened and doesn't transfer it some other way.

## 3. Uses what one step returns in the next

Type: `Create a Data Engineer job in Singapore needing Python and SQL, with 3 to 5 years of experience, then add Spark to it.`

- Jeni creates the job, reads the new id from VIRA's reply, and adds Spark to that id. Show it
  in the panel's feed: "Create job … → job 7105", then "Add job skills · job 7105 · Spark" (the
  id varies).

Point out:
- v1 can't do this in one request: its plan is made before the job exists, so it has no id to
  add the skill to.
- A new job card appears in the panel with Python, SQL and Spark.

## 4. Reads the data to decide, and the reviewer can change the call

Type: `Shortlist the two strongest applicants for the backend engineer job.`

- Jeni reads the applications and their match scores, and the card asks: "Shortlist
  applications 5102 and 5103." (0.91 and 0.85).
- Click **Change details**, set **App Ids** to `[5102]`, and click **Submit**. Say: "Only the top
  one for now."

Point out:
- The pick came from the data, not from a guess.
- What ran is the reviewer's version: in the panel only 5102 turns shortlisted. Jeni reports
  the edit. Shortlisting always comes back to a person: if Jeni tries 5103 again, that is a new
  card, and you reject it.

## 5. Asks only for what it can't look up, then carries on

Type: `Add Bob to the hiring team for the backend engineer job.`

- Jeni finds Bob (user 802) and the job on its own, then asks whether he joins as a team member
  or an administrator. That's the one thing it can't look up.

Type: `As a team member.`

- Jeni adds him with `role_id` 5.

Point out:
- v1 quietly defaults this to administrator. v2 has no default and asks.
- The answer continued the same task; nothing had to be retyped. The panel shows
  "Bob Tan · team member".

## 6. Never guesses personal data

Type: `Share application 5102 with the hiring manager.`

- Jeni asks for the hiring manager's email, and for a note to send with it.

Type: `Send it to priya.nair@example.com with a note: strong backend profile, worth a call.`

- The card asks: "Share application 5102 with priya.nair@example.com, with the note …".
  **Approve.**

Point out:
- Emails have to be exactly what the user typed. A guessed or copied address is refused in code,
  before VIRA, whatever the model tries.
- Candidate names and emails reach the model redacted; the feed shows "1 recipient", not the
  address.

## 7. Knows LinkedIn needs an open job, and does both steps when told

Type: `Publish the Data Analyst job to LinkedIn.`

- The Data Analyst job is closed. Usually Jeni asks whether it's open and public first, as the
  task catalog tells it to. If it tries anyway, VIRA refuses ("Job must be open and public before
  publishing to LinkedIn") and Jeni says so.

Type: `Reopen it, then publish it.`

- Jeni reopens the job, then publishes it.

Point out:
- The rule lives in VIRA, and Jeni reports it rather than working around it.
- The panel shows Data Analyst Open and On LinkedIn.

## 8. Says when something isn't supported

Type: `Share the Product Designer job on Facebook.`

- Jeni says sharing a job to Facebook isn't supported, and what it can do instead.

Point out:
- No approximation and no card: nothing reached VIRA.

## More to try (9–17)

Not in the presented run. Each works from the starting data, in a new chat: press **Reset
data** first. (After act 9 without a reset, "backend engineer" matches two jobs, and Jeni asks
which one you mean.)

### 9. Copies a job and changes only the copy

Type: `Copy the backend engineer job and add Rust to the copy only.`

- Jeni clones job 7001, reads the copy's new id (8001) and adds Rust to that id.
- The panel shows a second Senior Backend Engineer with Rust; job 7001 doesn't have it.

### 10. Makes two routine changes from one request

Type: `Remove PostgreSQL from the backend engineer job and make it public.`

- Two routine writes, no card: job 7001 loses PostgreSQL and turns public.

### 11. Edits a job's details

Type: `Set the Product Designer job to 2 openings, needing 2 to 4 years of experience.`

- One edit on job 7003. The feed shows "2–4 yrs · 2 openings".

### 12. Closes a job and records why

Type: `Close the Product Designer job: the role has been filled.`

- Job 7003 turns Closed, and the feed shows the reason.

### 13. Rejects an applicant once a person approves

Type: `Reject the weakest applicant for the backend engineer job.`

- Jeni reads the scores and the card asks to reject application 5104 (0.64). **Approve.**
- Rejecting, like shortlisting, always waits for a person.

### 14. Answers from the data without changing anything

Type: `Which applicant for the Data Analyst job has the best match score, and what stage are they at?`

- Jeni reads the applications and answers: 5201, 0.88, shortlisted. Nothing changes.

### 15. Finds suggested candidates for a job

Type: `Find suggested candidates for the backend engineer job.`

- Jeni lists the suggested candidates for job 7001 with their match scores. A read only.

### 16. Knows who "me" is

Type: `Share application 5103 with me, with a note: follow up next week.`

- The card says "Share application 5103 with you, with the note …". **Approve.** You are Sam
  Lee; the address is filled in only when the call is sent, and the model never sees it.
- A share needs a note; without one, Jeni asks for it.

### 17. Looks a colleague up and hands a job over, once approved

Type: `Transfer ownership of the Data Analyst job to Priya.`

- Jeni finds Priya in the company's users and the card asks to transfer job 7002 to her.
  **Approve.** The panel shows Priya Nair as the owner. Compare act 2, where you said no.

## Close (2 minutes)

- Scroll the "What reached VIRA" feed: every call the demo made, reads and writes, and nothing
  the reviewer rejected.
- What holds whatever the model does: high-stakes changes wait for a person (on real VIRA,
  every change does); ids must come from the user or an earlier result and keep their kind; personal values must be the user's words;
  inputs are bounded; results are masked before the model sees them; every call is audited.
- What's next: VIRA's task-group endpoint (Jeni's writes return 404 on real VIRA today), the
  approval design for production, and hosting with auth.

## Questions you may get

- **Which model?** gpt-5-mini (`CHAT_MODEL`). Any tool-calling model works; the guard doesn't
  depend on it.
- **Speed and cost?** 8–22 seconds per act including the approvals, about 145k tokens and about
  $0.05 for the eight presented acts (median of three `demo/rehearse.py` passes); all 17 took
  231 seconds and about $0.09 on 2026-10-05.
- **Is this real data?** No. The mock answers like VIRA and remembers changes for the demo. The
  tools go through the same guarded path as real mode.
- **Can it run on real VIRA?** The lookups can (TRON's database). The tasks wait on VIRA serving
  the task-group endpoint.
- **What if the model goes wrong?** High-stakes changes wait for approval here, and every change
  does on real VIRA. Whatever the mode, the guard refuses invented ids, ids of the wrong kind,
  emails the user didn't type, and repeats.

## If something goes wrong

- **A slow answer.** Up to 25 seconds is normal for acts 4 to 7; the dots show it's working.
- **The wrong job or person on a card.** Reject it and say why. Jeni stops.
- **A routine change went wrong.** Press **Reset data**, start a new chat and retype the prompt.
- **An unexpected question.** Answer it plainly, as you would a colleague.
- **An error, or nothing happens.** Start a new chat and retype the prompt. If the server is
  down, press Ctrl-C and run `demo/run.sh` again; the data starts over.
- **The data looks wrong.** Press **Reset data** in the panel.
- **As a backup,** screen-record a rehearsal beforehand.
