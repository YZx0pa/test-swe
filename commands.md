### find-talents
Find potential/suggested talents for a job (read-only).
Flags:
  --job-ids CSV     (REQUIRED) job ids to source talents for
  --profile-ids CSV (optional) specific profile ids to consider
Returns: profile_id values AND their scores (overall_score, skill_score, job_name_similarity).

### score-candidates
Trigger CV scoring calculation (composite + briq).
Flags:
  --app-ids CSV   (optional) application ids (applicants who applied)
  --match-ids CSV (optional) match ids (from get-match-id, for suggested talents)
Returns: scores[] with composite_score and briq per id.

### generate-jd
LLM-generate a job posting/description.
Flags:
  --job-title T   (REQUIRED) the job title
  --skills CSV    (optional) skills to include
  --lang C        (optional) language code, e.g. en, ar
  --job-id INT    (optional) existing job id, if any
Returns: job_description text.

### candidate-insights
Trigger candidate insights (v3) for applicants.
Flags:
  --app-ids CSV   (REQUIRED) application ids
  --match-ids CSV (optional) match ids
Returns: insights per id.
