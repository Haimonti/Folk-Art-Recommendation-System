#!/bin/bash
cd /opt/app-root/src
export PYTHONPATH=/opt/app-root/src:$PYTHONPATH
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
