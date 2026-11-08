# config/

Local configuration that stays out of git (everything here but this README is gitignored).

## `jeni_tasks.json`: Jeni's task catalog

`--tools jeni` (the LangGraph and deepagents runners, `compare_agents.py --suite jeni`) builds
one typed tool per task in this file. It is Jeni v1's `XAGENT_SUBTASK_API_CONFIG_SCHEMA` and is
shared internally, not through the repository.

- Put the shared file at `config/jeni_tasks.json`, or point `JENI_TASKS_FILE` (in `.env` or
  the shell) at it.
- Or regenerate it from v1's `tasks.js`:

  ```bash
  python jeni_tools.py --from-js path/to/tasks.js > config/jeni_tasks.json
  ```

- Check it: `python jeni_tools.py` lists every task (read or write) with its fields, and names
  any task the mock VIRA (`mock_jeni.py`) can't answer yet.

The offline tests don't need it: they use the synthetic `tests/fixtures/jeni_tasks.json`.

## `guard_tools.json`: the input guard's tool summary

The API's input guard (`input_guard.py`) describes Jeni's tools to its model in a short
summary made from the catalog above, saved here with a hash of the tool descriptions. After
the catalog changes, regenerate it and read it before use:

```bash
python -m input_guard            # up to date? (exit 1 when missing or stale)
python -m input_guard --refresh  # regenerate it with the chat model
```

While it is stale, the guard uses a plain one-line-per-tool list and logs
`INPUT_GUARD tool_summary_stale`.
