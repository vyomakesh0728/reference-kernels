-- Edit the target leaderboard name below when you want a different slice.
WITH target AS (
  SELECT id, name
  FROM leaderboards
  WHERE name = 'amd-fp8-mm'
),
ranked AS (
  SELECT
    s.submission_id,
    s.submission_time,
    s.run_mode,
    s.run_score,
    s.code
  FROM successful_submissions AS s
  JOIN target AS t
    ON t.id = s.leaderboard_id
  WHERE s.run_mode IN ('benchmark', 'leaderboard', 'test')
)
SELECT
  submission_id,
  submission_time,
  run_mode,
  run_score,
  LEFT(code, 2000) AS code_prefix
FROM ranked
ORDER BY submission_time DESC
LIMIT 50;
