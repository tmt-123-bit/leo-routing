#!/bin/bash
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy2.log 2>&1
echo "--- build tail:"
tail -3 /tmp/deploy2.log
echo "--- build errors:"
grep -E 'error' /tmp/deploy2.log | head -5
