#!/bin/zsh
# Kill all benchmark runs if OpenRouter spend crosses a ceiling.
# Usage: budget_guard.sh <ceiling_usd>
CEIL=${1:-40}
while true; do
  U=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['usage'])" 2>/dev/null)
  [ -z "$U" ] && sleep 60 && continue
  if python3 -c "import sys; sys.exit(0 if $U >= $CEIL else 1)"; then
    echo "BUDGET CEILING \$$CEIL HIT (spend \$$U) - stopping all runs"
    pkill -f "tau2 run"; pkill -f "taurl.run_sweep"
    exit 0
  fi
  pgrep -f "tau2 run" > /dev/null || pgrep -f "taurl.run_sweep" > /dev/null || { echo "runs finished; spend \$$U"; exit 0; }
  sleep 60
done
