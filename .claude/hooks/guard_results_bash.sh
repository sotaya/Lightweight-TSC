#!/bin/bash
cmd=$(/usr/bin/python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)
[ -z "$cmd" ] && exit 0

protected='(model\.h5|training\.log|best_params\.json|model\.pkl|output[0-9]*\.log|/runs(/|[[:space:]]|$))'
destructive='(^|[;&|`]|[[:space:]])(rm|rmdir|shred|truncate|mv)([[:space:]]|$)'

if echo "$cmd" | grep -qE "$destructive" && echo "$cmd" | grep -qE "$protected"; then
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"過去の実験結果(model.h5/training.log等)やrunsディレクトリの削除・移動は禁止されています。\"}}"
  exit 0
fi

if echo "$cmd" | grep -qE ">[^>]*$protected"; then
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"実験結果ファイルへの上書き(リダイレクト)は禁止されています。\"}}"
  exit 0
fi

exit 0
