FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY pipeline/ pipeline/
COPY mcp_server/ mcp_server/

ENV MCP_TRANSPORT=streamable-http
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "mcp_server/server.py"]
