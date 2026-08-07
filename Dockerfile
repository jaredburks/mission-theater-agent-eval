# Sandboxed execution: the agent loop runs inside the container, never on the host.
# This is the containerization story — extend it (gVisor/seccomp, network-off,
# read-only FS) to harden agent-authored tool execution.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir matplotlib
COPY . .
# Run the no-credential benchmark by default.
CMD ["python", "run_example.py"]
