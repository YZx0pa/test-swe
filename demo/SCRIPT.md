# Jeni v2 demo script

About 12 minutes: a 1-minute opening, eight short acts, a 2-minute close. Everything runs on
mock VIRA with synthetic jobs, people and applicants. Nothing reaches VIRA or a real database.

The prompts below are the ones `demo/rehearse.py` plays and checks, and the chat's starter cards
offer each act's first prompt. Type them as written, or click the card and press Enter. Start
each act in a **new chat** (the pencil icon, top right).

## Before the audience arrives

1. `demo/run.sh`, and wait for "Jeni v2 demo is up".
2. Rehearse once: `.venv/bin/python -m demo.rehearse` (about 2½ minutes, about $0.05). It should
   end `8/8`. It deletes its chats and resets the data when it finishes.
3. Open <http://localhost:3000>. Zoom the browser to 110–125% so the back row can read it. The
   data panel is on the right; **Show data** / **Hide data** toggles it.
4. For a business audience, switch on **Hide Tool Calls** under the input box. For engineers,
   leave the calls showing: they are the point.
5. If anything looks off, press **Reset data** in the panel. Every act works from the starting
   data.

Starting data, for your reference:

| Job | Id | State | Applicants (match score) |
|---|---|---|---|
| Senior Backend Engineer | 7001 | open, private | 5102 (0.91), 5103 (0.85), 5101 (0.72), 5104 (0.64) |
| Data Analyst | 7002 | closed, public | 5201 (0.88, shortlisted), 5202 (0.59) |
| Product Designer | 7003 | open, public | none |

Colleagues: Alice Johnson (801), Bob Tan (802), Priya Nair (803).

## Opening (1 minute)

- Jeni v1 plans a whole request in one model call, up front, and hands the plan to VIRA. It
  can't use what one step returns in the next, and it fills in defaults nobody chose.
- v2 works one step at a time. It reads real results before deciding the next step, and it asks
  a person before anything changes.
- Left: the chat. Right: the data. Watch the right side; it shows what really changed.

## 1. Finds the job by name, and asks before it changes anything

Type: `Add Kubernetes and Terraform to the backend engineer job.`

- Jeni looks up "backend engineer" (a read, so it runs straight away) and finds job 7001.
- An approval card appears for `add_job_skills`. **Approve.**

Point out:
- Nobody typed a job id. Lookups run on their own; anything that changes data waits for a person.
- In the panel, Senior Backend Engineer's skills gain Kubernetes and Terraform, and "What reached
  VIRA" shows the one write.

## 2. A person can say no

Type: `Close the Product Designer job, we've filled it.`

- A card appears for `make_job_closed` on job 7003. **Reject**, with the reason
  `Not yet: the hiring manager is still interviewing.`

Point out:
- Nothing reached VIRA: the feed has no new entry and the job is still Open.
- Jeni reports what happened and doesn't close it some other way.

## 3. Uses what one step returns in the next

Type: `Create a Data Engineer job in Singapore needing Python and SQL, with 3 to 5 years of experience, then add Spark to it.`

- Card 1, `create_job`: **Approve.** VIRA returns the new job's id.
- Card 2, `add_job_skills`: note that its `job_id` is that new id. **Approve.**

Point out:
- v1 can't do this in one request: its plan is made before the job exists, so it has no id to
  add the skill to.
- A new job card appears in the panel with Python, SQL and Spark.

## 4. Reads the data to decide, and the reviewer can change the call

Type: `Shortlist the two strongest applicants for the backend engineer job.`

- Jeni reads the applications and their match scores, then proposes
  `shortlist_multiple_application` for 5102 and 5103 (0.91 and 0.85).
- At the card, **edit** `app_ids` to `[5102]` and submit. Say: "Only the top one for now."

Point out:
- The pick came from the data, not from a guess.
- What ran is the reviewer's version: in the panel only 5102 turns shortlisted. Jeni reports
  the edit and doesn't go around it: until you write again, a new call to shortlist gets no card
  and is refused in code. If it offers to shortlist 5103 as well, that's only an offer.

## 5. Asks only for what it can't look up, then carries on

Type: `Add Bob to the hiring team for the backend engineer job.`

- Jeni finds Bob (user 802) and the job on its own, then asks whether he joins as a team member
  or an administrator. That's the one thing it can't look up.

Type: `As a team member.`

- Card for `add_job_collaborators` with `role_id` 5. **Approve.**

Point out:
- v1 quietly defaults this to administrator. v2 has no default and asks.
- The answer continued the same task; nothing had to be retyped. The panel shows
  "Bob Tan · team member".

## 6. Never guesses personal data

Type: `Share application 5102 with the hiring manager.`

- Jeni asks for the hiring manager's email, and for a note to send with it.

Type: `Send it to priya.nair@example.com with a note: strong backend profile, worth a call.`

- Card for `share_application`. **Approve.**

Point out:
- Emails have to be exactly what the user typed. A guessed or copied address is refused in code,
  before VIRA, whatever the model tries.
- Candidate names and emails reach the model redacted; the feed shows "1 recipient", not the
  address.

## 7. Knows LinkedIn needs an open job, and does both steps when told

Type: `Publish the Data Analyst job to LinkedIn.`

- The Data Analyst job is closed. Usually Jeni asks whether it's open and public first, as the
  task catalog tells it to. If it tries anyway, **approve**: VIRA refuses ("Job must be open and
  public before publishing to LinkedIn") and Jeni says so.

Type: `Reopen it, then publish it.`

- Card for `make_job_open`: **Approve.** Card for `publish_job_to_linkedin`: **Approve.**

Point out:
- The rule lives in VIRA, and Jeni reports it rather than working around it.
- The panel shows Data Analyst Open and On LinkedIn.

## 8. Says when something isn't supported

Type: `Share the Product Designer job on Facebook.`

- Jeni says sharing a job to Facebook isn't supported, and what it can do instead.

Point out:
- No approximation and no card: nothing reached VIRA.

## Close (2 minutes)

- Scroll the "What reached VIRA" feed: every call the demo made, reads and writes, and nothing
  the reviewer rejected.
- What holds whatever the model does: every write waits for a person; ids must come from the
  user or an earlier result and keep their kind; personal values must be the user's words;
  inputs are bounded; results are masked before the model sees them; every call is audited.
- What's next: VIRA's task-group endpoint (Jeni's writes return 404 on real VIRA today), the
  approval design for production, and hosting with auth.

## Questions you may get

- **Which model?** gpt-5-mini (`CHAT_MODEL`). Any tool-calling model works; the guard doesn't
  depend on it.
- **Speed and cost?** 8–30 seconds per act including the approvals, about 135k tokens and about
  $0.05 for the whole script (median of three `demo/rehearse.py` passes).
- **Is this real data?** No. The mock answers like VIRA and remembers changes for the demo. The
  tools go through the same guarded path as real mode.
- **Can it run on real VIRA?** The lookups can (TRON's database). The tasks wait on VIRA serving
  the task-group endpoint.
- **What if the model goes wrong?** It can only propose. Writes wait for approval, and the guard
  refuses invented ids, ids of the wrong kind, emails the user didn't type, and repeats.

## If something goes wrong

- **A slow answer.** Up to 30 seconds is normal for acts 4, 5 and 7; the dots show it's working.
- **The wrong job or person on a card.** Reject it and say why. Jeni stops.
- **An unexpected question.** Answer it plainly, as you would a colleague.
- **An error, or nothing happens.** Start a new chat and retype the prompt. If the server is
  down, press Ctrl-C and run `demo/run.sh` again; the data starts over.
- **The data looks wrong.** Press **Reset data** in the panel.
- **As a backup,** screen-record a rehearsal beforehand.
