#!/bin/bash
set -e

# Fix permissions on mounted volumes AFTER they are mounted at runtime
chmod -R 777 /app/chroma_db 2>/dev/null || true
chmod -R 777 /app/static 2>/dev/null || true

echo "Permissions set. Starting uvicorn..."

# Start without --reload to prevent SQLite WAL file conflicts
exec uvicorn app.main:app --host 0.0.0.0 --port 2000
