# Jeni entity DB-validation map

This is the validation plan for Jeni action payloads.  It is deliberately
separate from `grounding.py`: grounding proves a value came from the user or a
trusted prior tool result; this layer proves the referenced database entity is
real and tenant-scoped.

## Validation order

```text
LLM gathers values with Jeni read tools
→ Jeni action schema/Pydantic validates required fields and types
→ entity_db_validation (this map)
→ grounding / ToolCallGuard provenance check
→ VIRA
```

For `ALWAYS_CONFIRM` actions, validate once before the approval card and again
after a reviewer edit, before grounding and VIRA.  A validation failure returns
a structured tool error to the agent; it never calls VIRA.

## Composition rule: reusable entity lookups, not one DB helper per task

Do not create `validate_add_job_skills`, `validate_edit_job`, and so on.
Create one lookup per entity and compose those lookups by task profile:

```text
get_job_detail(job_id)                -> existence, tenant, canonical title, state
find_eligible_users(ids/emails)       -> active role-4 company users
find_candidate_by_email(email)        -> company-scoped candidate profile
validate_app_ids(app_ids)             -> tenant-scoped applications
```

For example, both `add_job_skills` and `edit_job` use the same `job_target`
profile and make one `get_job_detail(job_id)` call.  If a user wrote an
existing-job title as well as an ID, that one result also performs the title/ID
comparison.  The task determines which reusable checks are composed; it does
not determine a new SQL function.

## Field map

| Jeni field | Payload type | DB mapping / query | Rule | Status |
|---|---:|---|---|---|
| `job_id` | `int` | `hris.job.job_id`, scoped by `job.recuiter_company_id`, excluding `from_resume IS TRUE` | The job must exist in the authenticated company. | Existing `get_job_detail` / job-ID query |
| job title used as an existing-job reference | `str` metadata, not a VIRA field | `hris.jobname.name_name`, joined by `job.name_id` | Normalise; pass when exact, or canonical DB title contains the user phrase. Otherwise return `job_title_id_mismatch` to the agent. | Add source-reference metadata + comparison |
| `job_title` for `create_job` / `edit_job` | `str` | Same DB title column, but not used for validation | New value: do not compare it with the current title. Duplicate-title policy is optional and out of scope. | Skip |
| `app_ids` | `list[int]` | `hris.application.app_id`, joined to `hris.job` by `application.job_id` | Every requested app must exist and belong to the authenticated company. If a future payload also has `job_id`, require every app to belong to that job. | Existing app-ID query |
| `app_id` in read-only task | `int` | Same as above | Let the normal Jeni read task return not-found; no middleware check is needed for a read-only call. | Skip |
| `user_ids` | `list[int]` | `hris.userinfo.user_id`, `company_id`, `active`, `role_id` | Every requested user must exist, belong to company, be active, and have account `role_id = 4` (eligible to access the job). | **Add exact-ID mode to existing user query** |
| collaborator `role_id` | `int` | No confirmed collaborator-role table | User selects the job-assignment role. Allow only `{1, 5}` (`administrator`, `team member`). Do not compare it with `userinfo.role_id`. | Add allow-list schema/rule |
| `new_owner_user_email` | `str` | `hris.userinfo.email`, `company_id`, `role_id = 4`, `active IS TRUE` | Resolve token/`me`, then require one active role-4 internal user in company. | Add internal-email validator |
| `emails` for `share_application` | `list[str]` | Same `hris.userinfo` query as owner email | Resolve token/`me`; every recipient must be an active role-4 internal user in company. External recipients are not allowed by this policy. | Add internal-email validator |
| `candidate_email` | `str` | `hris.profile.email`; `profile.profile_id = application.profile_id`; application joined to job for company scope | Require a matching candidate profile with an application in the authenticated company. Compare emails case-insensitively. | **Add candidate-email query** |
| `candidate_name` | `str` | Candidate profile name columns not yet confirmed | Once known, compare the canonical profile name with the name supplied alongside `candidate_email`. | Pending DB column confirmation |
| `create_application_to_job` duplicate | relationship | `profile.email → application.profile_id`, with target `application.job_id` | Decide whether the same candidate may apply to the same job twice. If not, block an existing candidate+job relationship. | Confirm VIRA business rule |
| `is_private` | `bool` | `hris.job.is_private` | The task itself fixes the requested value; validate job existence only. | No extra check |
| `skills` | `list[str]` | No skills taxonomy supplied | No DB validation. | Skip |
| `job_description`, `job_requirements`, `reason_for_closure`, `message` | `str` | Not entity references | No DB validation. | Skip |
| `country_name`, `region` | `str` | No confirmed reference table | No DB validation. | Skip |
| `min_exp`, `max_exp`, `min_salary`, `max_salary`, `vacancy` | `int` | Not entity references | No DB lookup. Add deterministic range checks separately. | Add local rule |
| `search_key`, `limit`, `skip` | `str`, `int`, `int` | Read-tool inputs | No entity DB validation. Add local bounds checks for pagination. | Skip / local rule |

## Deterministic rules outside DB validation

```text
min_exp <= max_exp            when both are supplied
min_salary <= max_salary      when both are supplied
vacancy > 0                   when supplied
limit > 0 and skip >= 0       for read tools
role_id in {1, 5}             for add_job_collaborators
```

Optional state rules are deferred until VIRA behaviour is confirmed:

```text
publish_job_to_linkedin: job is open and public
make_job_open: currently closed
make_job_closed: currently open
```

## Required additions to current DB-related files

### `db_lookup.py`

Add direct, parameterised helpers.  These are internal validation helpers,
not LLM-facing tools.

1. Reuse and extend the existing `handle_search_users` base query.  It already
defines the eligible internal-user population (`company_id`, `role_id = 4`,
`active IS TRUE`).  Add optional exact filters rather than new user SQL:

```text
xuser_ids  -> AND ui.user_id = ANY($n::bigint[])
xemails    -> AND lower(ui.email) = ANY($n::text[])
```

The existing `search_keys` remains a partial name/email search for the LLM;
the validation middleware uses only the exact-ID or exact-email modes.

```sql
SELECT ui.user_id, ui.firstname, ui.lastname, ui.email
FROM hris.userinfo ui
WHERE ui.company_id = $1
  AND ui.role_id = 4
  AND ui.active IS TRUE;
```

Use that same helper for both `user_ids` and the exact email checks for
`new_owner_user_email` / `share_application.emails`.

3. `handle_find_candidate_by_email(conn, company_id, email)`:

```sql
SELECT DISTINCT p.profile_id, p.email
FROM hris.profile p
JOIN hris.application a ON a.profile_id = p.profile_id
JOIN hris.job j ON j.job_id = a.job_id
WHERE j.recuiter_company_id = $1
  AND lower(p.email) = lower($2)
LIMIT 2;
```

Add the candidate-name columns only after confirming their schema.  Do not
assume they are `firstname` / `lastname`.

### `db_queries.py`

- Retain `get_job_detail`; it already returns canonical `job_title`,
  `is_private`, and derived `is_open` for middleware use.
- Add QueryTool contracts for `validate_user_ids`, `validate_internal_emails`
  (both using the extended existing user helper), and
  `find_candidate_by_email`, plus matching fake implementations.
- Keep these contracts available to the new validation middleware.  They do
  not need to be exposed to the LLM.

### `db_tools.py`

- The duplicated information reads (`get_job_detail`, `find_user`, and
  `list_job_applications`) have already been removed from the LLM tool list;
  Jeni's `get_single_job_details`, `search_users`, and `get_applications`
  should be used instead.
- After the middleware is live, remove all `validate_*` tools from
  `LLM_TOOL_NAMES` and delete the prompt instruction telling the LLM to call
  them.  The only current direct DB discovery tool the LLM needs is
  `find_job_by_title`.

### New `entity_db_validation.py`

Implement the middleware/helper that maps `(tool_name, normalised_args)` to
the rules above.  It should:

1. use `auth_profile.company_id`, never a model-provided company ID;
2. compare requested `list[int]` values against returned row-ID sets;
3. resolve email tokens/`me` before a case-insensitive comparison copy;
4. return structured `validation_error` results to the agent on mismatch;
5. log `ENTITY_DB_VALIDATION_SKIPPED` and allow only when a task has no rule;
6. treat DB/query failures as `validation_unavailable` and block a real write.

### `agent_kit.py`

Wire entity validation into the Jeni action-tool middleware path before
`ToolCallGuard` grounding.  For `ALWAYS_CONFIRM` calls, run it before the
approval card and again after an edited approval payload.

`ToolCallGuard` continues to use `grounding.py` for provenance only:
user-written or trusted earlier-result IDs, correct ID kind, and user-only
fields.  It does not determine database existence.
