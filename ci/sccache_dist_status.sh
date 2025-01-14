#!/usr/bin/env bash

set -euo pipefail

# Print current dist status to verify we're connected
sccache --dist-status | jq -r -f <(cat <<"EOF"
  .SchedulerStatus as [$x, $y] | [
    ($y | {type: "scheduler", id: $x, servers: (.servers | length), cpus: .info.occupancy, util: ((.info.cpu_usage // 0) * 100 | round | . / 100 | tostring | . + "%"), jobs: (.jobs.fetched + .jobs.running), fetched: .jobs.fetched, running: .jobs.running, u_time: ($y.servers // {} | map(.u_time) | min | . // 0 | tostring | . + "s")}),
    ($y.servers // {}
      | to_entries
      | sort_by(.key)
      | map(
        .key as $k
        | .value
        | {type: "server", id: .id, servers: 1, cpus: .info.occupancy, util: ((.info.cpu_usage // 0) * 100 | round | . / 100 | tostring | . + "%"), jobs: (.jobs.fetched + .jobs.running), fetched: .jobs.fetched, running: .jobs.running, u_time: ((.u_time // 0) | tostring | . + "s")}
      )[]
    )
  ] as $rows
  | ($rows[0] | keys_unsorted) as $cols
  | ($rows | map(. as $row | $cols | map($row[.]))) as $rows
  | $cols, $rows[] | @csv | gsub("\""; "")
EOF
) | column -t -s, -R 1-7
