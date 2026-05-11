#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/new-day.sh NNN topic-slug" >&2
  exit 1
fi

DAY="$1"
TOPIC="$2"
DATE="$(date +%Y-%m-%d)"
FILE="days/${DATE}-day-${DAY}-${TOPIC}.md"

if [[ -e "$FILE" ]]; then
  echo "Already exists: $FILE" >&2
  exit 1
fi

cp templates/day.md "$FILE"
sed -i.bak "s/Day NNN - Topic/Day ${DAY} - ${TOPIC}/" "$FILE"
sed -i.bak "s/Date: YYYY-MM-DD/Date: ${DATE}/" "$FILE"
rm "${FILE}.bak"

echo "$FILE"

