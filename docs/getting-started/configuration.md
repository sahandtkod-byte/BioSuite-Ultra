# Configuration

BioSuite uses environment variables for configuration. No secrets are stored in code.

## Required Environment Variables (v5.0.0+)

```bash
# API authentication
export BIOSUITE_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Admin credentials
export BIOSUITE_ADMIN_USER=admin
export BIOSUITE_ADMIN_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
```

## Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOSUITE_LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `BIOSUITE_MAX_REQUEST_SIZE_MB` | `50` | Max API request size in MB |
| `BIOSUITE_MAX_SEQUENCE_LENGTH` | `10000000` | Max sequence length for API |
| `BIOSUITE_JWT_EXPIRE_SECONDS` | `3600` | JWT token expiration |
| `BIOSUITE_JWT_SECRET` | auto-generated | JWT signing secret |
| `BIOSUITE_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Config File

BioSuite stores config in `~/.biosuite/biosuite_config.json`:

```json
{
  "theme": "dark-green",
  "default_dpi": 180,
  "save_format": "png",
  "interactive": false,
  "downsample_threshold": 5000,
  "quiet": true
}
```

## Docker Configuration

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - BIOSUITE_API_KEY=your-key
      - BIOSUITE_ADMIN_USER=admin
      - BIOSUITE_ADMIN_PASSWORD=your-password
      - BIOSUITE_LOG_LEVEL=INFO
```
