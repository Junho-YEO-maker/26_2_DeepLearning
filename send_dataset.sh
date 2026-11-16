#!/usr/bin/env bash
# Send local Dataset folder to junho:~/26_2_DeepLearning/Dataset (one password prompt)
set -uo pipefail

SRC_PARENT="/c/Users/asap0/OneDrive/바탕 화면/yonsei/26_2_DeepLearning"
cd "$SRC_PARENT" || { echo "[local] folder not found"; exit 1; }

echo "[local] Dataset: 11,522 files, about 3.2 GB"
echo "[local] Enter the server password when asked. Transfer has no progress bar; wait for DONE."
start=$(date +%s)

tar -cf - Dataset | ssh junho '
  mkdir -p ~/26_2_DeepLearning && cd ~/26_2_DeepLearning || exit 2
  if [ -n "$(ls -A Dataset 2>/dev/null)" ]; then
    echo "[server] ~/26_2_DeepLearning/Dataset already has files. Stopped without overwriting."
    ls -la Dataset
    exit 3
  fi
  tar -xf - || exit 4
  echo "[server] files: $(find Dataset -type f | wc -l)"
  echo "[server] size:  $(du -sh Dataset | cut -f1)"
  ls -la Dataset
'
rc=$?
echo "[local] elapsed: $(( $(date +%s) - start )) s, exit code: $rc"
[ $rc -eq 0 ] && echo "DONE" || echo "FAILED"
