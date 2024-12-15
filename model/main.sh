#!/bin/bash

# CPU 코어 수를 확인
CORES=$(nproc)

# 권장되는 worker 수는 (2 x CPU 코어 수) + 1
WORKERS=$((2 * CORES + 1))

# uvicorn 실행
uvicorn main:app \
    --host "0.0.0.0" \
    --port 8001 \
    --workers $WORKERS \
    --log-level info
