# Mission Theater — Benchmark Report

Episodes: 60  |  Agents: random, greedy, llama3.1-8b, qwen3-14b, gemma4

## Leaderboard

| agent | mean_composite | solve_rate | mean_completion_pct | mean_steps | mean_hazard_hits | mean_units_lost | episodes |
|---|---|---|---|---|---|---|---|
| greedy | 65.97 | 0.333 | 64.4 | 37.4 | 0 | 1 | 12 |
| qwen3-14b | 27.72 | 0.0 | 16.9 | 37.3 | 0.92 | 1.83 | 12 |
| gemma4 | 7.22 | 0.0 | 4.5 | 29.5 | 1.25 | 2.25 | 12 |
| random | 3.56 | 0.0 | 0.0 | 39 | 0.17 | 1.17 | 12 |
| llama3.1-8b | 1.11 | 0.0 | 0.0 | 38.2 | 0.67 | 1.5 | 12 |

## Failure modes

- **random** — incomplete_other: 11, hazard_death: 1, looping: 1
- **greedy** — incomplete_other: 6, solved: 4, ignored_interrupt: 2
- **llama3.1-8b** — under_scanned: 12, ignored_interrupt: 12, looping: 12, hazard_death: 5, early_termination: 1
- **qwen3-14b** — hazard_death: 8, ignored_interrupt: 7, early_termination: 1
- **gemma4** — hazard_death: 9, looping: 7, early_termination: 5, ignored_interrupt: 3, under_scanned: 1
