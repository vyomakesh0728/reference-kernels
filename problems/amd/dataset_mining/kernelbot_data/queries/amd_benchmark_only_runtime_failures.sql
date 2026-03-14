WITH amd_leaderboards AS (
  SELECT id, name
  FROM leaderboards
  WHERE name LIKE 'amd-%'
),
per_code AS (
  SELECT
    s.leaderboard_id,
    l.name AS leaderboard_name,
    s.code_id,
    MAX(CASE WHEN s.run_mode = 'test' AND COALESCE(s.run_passed, FALSE) THEN 1 ELSE 0 END) AS has_passing_test,
    MAX(CASE WHEN s.run_mode = 'benchmark' AND NOT COALESCE(s.run_passed, FALSE) THEN 1 ELSE 0 END) AS has_failing_benchmark,
    MAX(CASE WHEN s.run_mode = 'benchmark' AND NOT COALESCE(s.run_passed, FALSE) THEN s.submission_time ELSE NULL END) AS latest_benchmark_failure_time
  FROM submissions AS s
  JOIN amd_leaderboards AS l
    ON l.id = s.leaderboard_id
  GROUP BY 1, 2, 3
)
SELECT
  leaderboard_id,
  leaderboard_name,
  COUNT(*) AS code_variants
FROM per_code
WHERE has_passing_test = 1
  AND has_failing_benchmark = 1
GROUP BY 1, 2
ORDER BY code_variants DESC, leaderboard_id;
