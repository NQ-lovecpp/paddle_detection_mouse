#!/bin/bash
# 等待 C 组全部完成后，自动启动 Y6
BASE=/hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

echo "[$(date '+%H:%M:%S')] 等待 C 组训练完成..."
while true; do
  C1_DONE=$([ -f "$BASE/output/C1_picodet_s_qat/DONE" ] && echo 1 || echo 0)
  C2_DONE=$([ -f "$BASE/output/C2_yolov3_fpgm_prune/DONE" ] && echo 1 || echo 0)
  C3_DONE=$([ -f "$BASE/output/C3_yolov3_distill/DONE" ] && echo 1 || echo 0)
  echo "[$(date '+%H:%M:%S')] C1=$C1_DONE  C2=$C2_DONE  C3=$C3_DONE"
  if [ "$C1_DONE" = "1" ] && [ "$C2_DONE" = "1" ] && [ "$C3_DONE" = "1" ]; then
    echo "[$(date '+%H:%M:%S')] C 组全部完成！启动 Y6..."
    break
  fi
  sleep 300  # 每 5 分钟检查一次
done

cd $BASE
python scripts/run_experiments_round2.py --groups B --from B3 \
  >> output/round2_b_launcher.log 2>&1
