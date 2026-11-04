#!/bin/bash
fp=$(/usr/bin/python3 -I -c 'import json,sys; d=json.load(sys.stdin).get("tool_input",{}); print(d.get("file_path",""))' 2>/dev/null)
[ -z "$fp" ] && exit 0

if echo "$fp" | grep -qE '(model\.h5|training\.log|best_params\.json|model\.pkl|output[0-9]*\.log)$'; then
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"過去の実験結果ファイルの上書きは禁止されています。\"}}"
  exit 0
fi

exit 0
