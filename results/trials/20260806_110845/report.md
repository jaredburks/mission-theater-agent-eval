# Mission Theater — Benchmark Report

Episodes: 36  |  Agents: random, greedy, claude

## Leaderboard

| agent | mean_composite | solve_rate | mean_completion_pct | mean_steps | mean_hazard_hits | mean_units_lost | episodes |
|---|---|---|---|---|---|---|---|
| claude | 67.32 | 0.25 | 64.1 | 37.2 | 1 | 1.92 | 12 |
| greedy | 65.97 | 0.333 | 64.4 | 37.4 | 0 | 1 | 12 |
| random | 3.56 | 0.0 | 0.0 | 39 | 0.17 | 1.17 | 12 |

## Failure modes

- **random** — incomplete_other: 11, hazard_death: 1, looping: 1
- **greedy** — incomplete_other: 6, solved: 4, ignored_interrupt: 2
- **claude** — hazard_death: 8, ignored_interrupt: 5, solved: 3, looping: 2, incomplete_other: 1
