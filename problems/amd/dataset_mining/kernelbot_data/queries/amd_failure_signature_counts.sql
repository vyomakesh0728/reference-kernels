WITH amd_leaderboards AS (
  SELECT id, name
  FROM leaderboards
  WHERE name LIKE 'amd-%'
),
amd_fails AS (
  SELECT
    s.leaderboard_id,
    l.name AS leaderboard_name,
    s.run_mode,
    COALESCE(CAST(s.run_result AS VARCHAR), '') AS run_result_text
  FROM submissions AS s
  JOIN amd_leaderboards AS l
    ON l.id = s.leaderboard_id
  WHERE COALESCE(s.run_passed, FALSE) = FALSE
    AND s.run_mode IN ('benchmark', 'leaderboard', 'profile', 'test')
),
normalized AS (
  SELECT
    leaderboard_id,
    leaderboard_name,
    run_mode,
    CASE
      WHEN lower(run_result_text) LIKE '%memory access fault%' THEN 'memory_access_fault'
      WHEN lower(run_result_text) LIKE '%read-only page%' THEN 'write_to_read_only_page'
      WHEN lower(run_result_text) LIKE '%illegal%' THEN 'illegal_access_or_instruction'
      WHEN lower(run_result_text) LIKE '%timed out%' THEN 'timeout'
      WHEN lower(run_result_text) LIKE '%out of memory%' THEN 'oom'
      WHEN lower(run_result_text) LIKE '%runtime error%' THEN 'runtime_error'
      ELSE 'other'
    END AS failure_signature
  FROM amd_fails
)
SELECT
  leaderboard_id,
  leaderboard_name,
  run_mode,
  failure_signature,
  COUNT(*) AS failure_count
FROM normalized
GROUP BY 1, 2, 3, 4
ORDER BY failure_count DESC, leaderboard_id, run_mode, failure_signature;
