WITH amd_leaderboards AS (
  SELECT id, name
  FROM leaderboards
  WHERE name LIKE 'amd-%'
)
SELECT
  s.submission_id,
  s.leaderboard_id,
  l.name AS leaderboard_name,
  s.submission_time,
  s.run_mode,
  LEFT(CAST(s.run_result AS VARCHAR), 500) AS result_snippet
FROM submissions AS s
JOIN amd_leaderboards AS l
  ON l.id = s.leaderboard_id
WHERE COALESCE(s.run_passed, FALSE) = FALSE
  AND (
    lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%memory access fault%'
    OR lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%read-only page%'
    OR lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%illegal%'
  )
ORDER BY s.submission_time DESC
LIMIT 200;
