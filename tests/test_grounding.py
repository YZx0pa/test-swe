"""grounding: every tool argument and answer number traced to the task or an earlier result."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import grounding

TOP_PICK = ("Score applicants 11, 12 and 13, then get candidate insights only for the one "
            "with the highest composite score.")
SCORES = json.dumps({"status": "ok", "result": {"scores": [
    {"app_id": 11, "composite_score": 0.78}, {"app_id": 12, "composite_score": 0.95},
    {"app_id": 13, "composite_score": 0.71}]}})


def top_pick_messages(answer="Applicant 12 scored highest (0.95): strong backend fit."):
    return [
        HumanMessage(TOP_PICK),
        AIMessage("", tool_calls=[{"name": "score_candidates", "args": {"app_ids": [11, 12, 13]},
                                   "id": "c1", "type": "tool_call"}]),
        ToolMessage(SCORES, tool_call_id="c1", name="score_candidates"),
        AIMessage("", tool_calls=[{"name": "candidate_insights", "args": {"app_ids": [12]},
                                   "id": "c2", "type": "tool_call"}]),
        ToolMessage('{"status": "ok", "result": {"insights": [{"app_id": 12, "summary": '
                    '"strong backend fit"}]}}', tool_call_id="c2", name="candidate_insights"),
        AIMessage(answer),
    ]


def test_values_are_traced_to_the_task_and_to_earlier_results():
    steps = grounding.trace_from_messages(top_pick_messages(), TOP_PICK)
    assert [s["kind"] for s in steps] == ["call", "result", "call", "result", "answer"]
    assert steps[0]["provenance"]["app_ids"] == [
        {"value": v, "sources": ["task"]} for v in (11, 12, 13)]
    # the pick: 12 is in the task, and in step 2's result, where it is the top score
    assert steps[2]["provenance"]["app_ids"] == [{"value": 12, "sources": ["task", "step 2"]}]
    answer = {n["value"]: n["sources"] for n in steps[4]["numbers"]}
    assert answer == {"12": ["task", "step 2", "step 4"], "0.95": ["step 2"]}
    assert grounding.summary(steps) == {"args": 4, "args_grounded": 4, "args_chained": 1,
                                        "numbers": 2, "numbers_grounded": 2, "ungrounded": []}


def test_invented_values_are_flagged():
    task = "Write an Arabic job description for a Senior Backend Engineer with Python and Go skills."
    messages = [
        HumanMessage(task),
        AIMessage("", tool_calls=[{"name": "generate_jd", "type": "tool_call", "id": "c1",
                                   "args": {"job_title": "Senior Backend Engineer", "lang": "ar",
                                            "skills": ["Python", "Go", "Kubernetes"],
                                            "other_requirements": ["5+ years experience"]}}]),
        ToolMessage('{"status": "ok", "result": {"job_description": "[ar] Draft JD"}}',
                    tool_call_id="c1", name="generate_jd"),
        AIMessage("Done. Requires 5+ years; match score 0.99."),
    ]
    steps = grounding.trace_from_messages(messages, task)
    prov = steps[0]["provenance"]
    assert prov["lang"] == [{"value": "ar", "sources": ["task"]}]          # "Arabic" in the task
    assert [p["sources"] for p in prov["skills"]] == [["task"], ["task"], []]
    assert prov["other_requirements"] == [{"value": "5+ years experience", "sources": []}]
    assert grounding.summary(steps)["ungrounded"] == ["Kubernetes", "5+ years experience",
                                                      "5+", "0.99"]


def test_english_is_a_default_and_percentages_match_scores():
    sources = [("task", "Draft a JD for a Data Analyst"), ("step 2", '{"composite_score": 0.95}')]
    assert grounding.ground_args({"lang": "en"}, sources) == {
        "lang": [{"value": "en", "sources": ["default"]}]}
    assert grounding.ground_answer("Scored 95%.", sources) == [
        {"value": "95%", "sources": ["step 2"]}]


def test_refusals_are_marked():
    messages = [HumanMessage("x"),
                AIMessage("", tool_calls=[{"name": "find_talents", "args": {"job_ids": [1]},
                                           "id": "c1", "type": "tool_call"}]),
                ToolMessage('{"status": "error", "message": "Refused: identical"}',
                            tool_call_id="c1", name="find_talents", status="error"),
                AIMessage("ok")]
    assert grounding.trace_from_messages(messages, "x")[1]["refused"] is True


def _mini_call(command):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"arguments": json.dumps({"command": command})}}]}


def _mini_output(text, code=0):
    return {"role": "tool", "content": json.dumps({"returncode": code, "output": text})}


def test_mini_commands_are_parsed_and_off_policy_output_hidden():
    task = "Find talents for job 123 and score them."
    messages = [
        _mini_call("python3 recruiter_cli.py --mode mock find-talents --job-ids 123"),
        _mini_output('{"status": "ok", "result": {"suggested_profiles": [{"profile_id": 900001}]}}'),
        _mini_call("cat .env"),
        _mini_output("SECRET=do-not-show"),
        _mini_call('echo "SUMMARY: found 900001 | no match_ids | cannot score"'),
        _mini_output("SUMMARY: found 900001 | no match_ids | cannot score"),
        _mini_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]
    steps = grounding.trace_from_mini(messages, task)
    kinds = [s["kind"] for s in steps]
    assert kinds == ["call", "result", "call", "result", "echo", "answer"]
    assert steps[0]["tool"] == "find_talents"
    assert steps[0]["provenance"] == {"job_ids": [{"value": 123, "sources": ["task"]}]}
    assert steps[2]["off_policy"] and "do-not-show" not in json.dumps(steps)
    assert steps[-1]["text"].startswith("SUMMARY") and steps[-1]["numbers"] == [
        {"value": "900001", "sources": ["step 2"]}]


def test_parse_cli_keeps_typos_as_ungroundable_strings():
    tool, args = grounding.parse_cli("python3 recruiter_cli.py --mode mock find-talents "
                                     "--job-ids 123..")
    assert tool == "find_talents" and args == {"job_ids": ["123.."]}
    assert grounding.ground_args(args, [("task", "job 123")]) == {
        "job_ids": [{"value": "123..", "sources": []}]}
