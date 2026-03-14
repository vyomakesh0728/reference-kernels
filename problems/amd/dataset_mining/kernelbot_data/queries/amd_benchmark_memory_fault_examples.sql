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
    MAX(CASE WHEN s.run_mode = 'benchmark' AND NOT COALESCE(s.run_passed, FALSE) THEN 1 ELSE 0 END) AS has_failing_benchmark
  FROM submissions AS s
  JOIN amd_leaderboards AS l
    ON l.id = s.leaderboard_id
  GROUP BY 1, 2, 3
),
benchmark_fails AS (
  SELECT
    s.submission_id,
    s.code_id,
    s.leaderboard_id,
    l.name AS leaderboard_name,
    s.submission_time,
    LEFT(CAST(s.run_result AS VARCHAR), 500) AS result_snippet,
    LEFT(s.code, 2000) AS code_prefix
  FROM submissions AS s
  JOIN amd_leaderboards AS l
    ON l.id = s.leaderboard_id
  JOIN per_code AS c
    ON c.leaderboard_id = s.leaderboard_id
   AND c.code_id = s.code_id
  WHERE c.has_passing_test = 1
    AND c.has_failing_benchmark = 1
    AND s.run_mode = 'benchmark'
    AND NOT COALESCE(s.run_passed, FALSE)
    AND (
      lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%memory access fault%'
      OR lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%read-only page%'
      OR lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%illegal%'
      OR lower(COALESCE(CAST(s.run_result AS VARCHAR), '')) LIKE '%timed out%'
    )
)
SELECT
  submission_id,
  leaderboard_id,
  leaderboard_name,
  submission_time,
  result_snippet,
  code_prefix
FROM benchmark_fails
ORDER BY submission_time DESC
LIMIT 50;
