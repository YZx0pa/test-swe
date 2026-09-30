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
