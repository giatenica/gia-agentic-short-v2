# Deployment Guide

This guide covers deploying the GIA Agentic Research System in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)

---

## Prerequisites

### Required Software

- Python 3.11+
- uv (package manager)
- Git

### Required API Keys

| Service | Description | Get Key |
|---------|-------------|---------|
| Anthropic | Claude LLM API | [console.anthropic.com](https://console.anthropic.com/) |
| LangSmith | Tracing and debugging | [smith.langchain.com](https://smith.langchain.com/) |
| Tavily | Web search API | [tavily.com](https://tavily.com/) |

---

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/giatenica/gia-agentic-short-v2.git
cd gia-agentic-short-v2

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run LangGraph Studio
cd studio && uv run langgraph dev

# Or run CLI
uv run python -m src.main
```

### Development Environment

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/
```

---

## Production Deployment

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB |
| Network | 10 Mbps | 100+ Mbps |

### Installation

```bash
# Create production directory
mkdir -p /opt/gia-agentic
cd /opt/gia-agentic

# Clone and setup
git clone https://github.com/giatenica/gia-agentic-short-v2.git .
uv sync --no-dev

# Configure environment
cp .env.example .env
# Edit .env for production
```

### Production Environment Variables

```bash
# .env for production
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_...
TAVILY_API_KEY=tvly-...

# Production settings
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=gia-agentic-prod

# Caching (enable for cost reduction)
CACHE_ENABLED=true
CACHE_PATH=/var/lib/gia-agentic/cache.db
CACHE_TTL_LITERATURE=7200
CACHE_TTL_SYNTHESIS=3600
CACHE_TTL_WRITER=1800

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Process Management (systemd)

Create `/etc/systemd/system/gia-agentic.service`:

```ini
[Unit]
Description=GIA Agentic Research System
After=network.target

[Service]
Type=simple
User=gia
Group=gia
WorkingDirectory=/opt/gia-agentic
Environment="PATH=/opt/gia-agentic/.venv/bin"
ExecStart=/opt/gia-agentic/.venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable gia-agentic
sudo systemctl start gia-agentic
sudo systemctl status gia-agentic
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev

# Copy application
COPY src/ ./src/
COPY studio/ ./studio/
COPY public/ ./public/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "src.main"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  gia-agentic:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - LANGSMITH_TRACING=true
      - CACHE_ENABLED=true
      - CACHE_PATH=/data/cache.db
    volumes:
      - gia-data:/data
    restart: unless-stopped

  # Optional: LangGraph Studio
  langgraph-studio:
    build:
      context: .
      dockerfile: Dockerfile.studio
    ports:
      - "2024:2024"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    depends_on:
      - gia-agentic

volumes:
  gia-data:
```

### Build and Run

```bash
# Build
docker compose build

# Run
docker compose up -d

# View logs
docker compose logs -f gia-agentic

# Stop
docker compose down
```

---

## Cloud Deployment

### AWS (EC2 + ECS)

#### EC2 Setup

```bash
# Launch Ubuntu 22.04 instance (t3.medium or larger)
# SSH into instance

# Install dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/giatenica/gia-agentic-short-v2.git
cd gia-agentic-short-v2
uv sync --no-dev

# Configure and run
cp .env.example .env
# Edit .env with your keys
```

#### ECS Task Definition

```json
{
  "family": "gia-agentic",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "gia-agentic",
      "image": "your-ecr-repo/gia-agentic:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "LANGSMITH_TRACING", "value": "true"},
        {"name": "CACHE_ENABLED", "value": "true"}
      ],
      "secrets": [
        {"name": "ANTHROPIC_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "LANGSMITH_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "TAVILY_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gia-agentic",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "gia"
        }
      }
    }
  ]
}
```

### Google Cloud (Cloud Run)

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: gia-agentic
spec:
  template:
    spec:
      containers:
        - image: gcr.io/your-project/gia-agentic:latest
          ports:
            - containerPort: 8000
          env:
            - name: LANGSMITH_TRACING
              value: "true"
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: anthropic
          resources:
            limits:
              memory: 2Gi
              cpu: "1"
```

Deploy:

```bash
gcloud run deploy gia-agentic \
  --image gcr.io/your-project/gia-agentic:latest \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated
```

---

## Environment Configuration

### Complete Environment Variables

```bash
# Required API Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
LANGSMITH_API_KEY=lsv2_pt_...
TAVILY_API_KEY=tvly-...

# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=gia-agentic-prod
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Model Configuration
DEFAULT_MODEL=claude-sonnet-4-5-20250929
COMPLEX_MODEL=claude-opus-4-5-20251101
FAST_MODEL=claude-haiku-4-5-20251001

# Caching Configuration
CACHE_ENABLED=true
CACHE_PATH=./data/cache.db
CACHE_TTL_DEFAULT=1800
CACHE_TTL_LITERATURE=3600
CACHE_TTL_SYNTHESIS=1800
CACHE_TTL_GAP_ANALYSIS=1800
CACHE_TTL_WRITER=600

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Flask Server (if using intake form)
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Data Directory
DATA_DIR=./data
UPLOAD_DIR=./uploads
```

### Configuration Precedence

1. Environment variables (highest priority)
2. `.env` file in project root
3. Default values in `src/config/settings.py`

---

## Monitoring and Logging

### LangSmith Integration

All workflow executions are automatically traced to LangSmith:

1. View traces at [smith.langchain.com](https://smith.langchain.com/)
2. Filter by project: `LANGSMITH_PROJECT`
3. Debug individual node executions
4. Monitor latency and costs

### Structured Logging

```python
import logging
import json

# Configure JSON logging for production
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "extra": getattr(record, "extra", {}),
        })

# Usage
logger.info("Workflow started", extra={"thread_id": "abc123"})
```

### Health Checks

```python
# Add to src/server.py
@app.route("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.route("/ready")
def ready():
    # Check dependencies
    checks = {
        "anthropic": check_anthropic_api(),
        "langsmith": check_langsmith_api(),
        "tavily": check_tavily_api(),
    }
    status = "ready" if all(checks.values()) else "not_ready"
    return {"status": status, "checks": checks}
```

---

## Security Considerations

### API Key Management

- **Never** commit API keys to version control
- Use environment variables or secret managers
- Rotate keys periodically
- Use separate keys for development and production

### Network Security

- Enable HTTPS in production
- Restrict CORS to trusted origins
- Use API rate limiting
- Monitor for unusual access patterns

### Data Protection

- Encrypt data at rest (SQLite cache, uploads)
- Sanitize user inputs
- Validate file uploads (type, size, content)
- Implement access controls for multi-tenant deployments

### Input Validation

- ZIP extraction protected against zip bombs
- Path traversal prevention
- Safe expression evaluation (no arbitrary code execution)
- SQL injection prevention in DuckDB queries

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| API key errors | Verify keys in `.env`, check for whitespace |
| Rate limiting | Enable caching, reduce concurrent requests |
| Memory issues | Increase container/instance memory |
| Slow execution | Check LangSmith traces for bottlenecks |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug
uv run python -m src.main --debug
```

### Support

- GitHub Issues: [github.com/giatenica/gia-agentic-short-v2/issues](https://github.com/giatenica/gia-agentic-short-v2/issues)
- Documentation: [docs/](docs/)
- LangSmith: [smith.langchain.com](https://smith.langchain.com/)

---

*Deployment Guide maintained by Gia Tenica. Last updated: January 2026.*
