# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 5.5.x   | ✅ Active support |
| 5.0.x   | ⚠️ Security fixes only |
| < 5.0   | ❌ End of life |

## Reporting a vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

### How to report

1. Email: **sahandtkod@gmail.com**
2. Subject: `[SECURITY] BioSuite Ultra - <brief description>`
3. Include: affected version, description, steps to reproduce, impact assessment

If you prefer a GitHub-native channel and private vulnerability reporting is enabled on the
repository, use **Security → Report a vulnerability** instead.

### What to expect

- **Acknowledgment**: within 48 hours
- **Assessment**: within 7 days
- **Fix timeline**: Critical (1 week), High (2 weeks), Medium (1 month)
- **Credit**: reporters will be credited in release notes (unless anonymity is preferred)

### Scope

- REST API authentication and authorization
- CLI input handling (injection, evaluation of untrusted input)
- File I/O (path traversal, arbitrary read/write)
- Dependency vulnerabilities
- Docker/container escape vectors

### Out of scope

- Social engineering attacks
- Denial of service against self-hosted instances
- Issues in third-party dependencies (please report those upstream)

## Credential handling

BioSuite Ultra is designed to fail closed: the REST API **refuses to start** unless its
credentials are supplied through the environment. There is no default admin password and no
fallback JWT signing secret.

| Variable | Purpose | Required |
| --- | --- | --- |
| `BIOSUITE_API_KEY` | Shared key required by every `/api/*` endpoint via the `X-API-Key` header | Yes |
| `BIOSUITE_JWT_SECRET` | Signing secret for admin bearer tokens | Yes |
| `BIOSUITE_ADMIN_PASSWORD` | Password for `/api/v1/admin/login` | Yes |
| `BIOSUITE_DATA_DIR` | Directory that file endpoints may read from | Recommended |
| `BIOSUITE_CORS_ORIGINS` | Comma-separated allowed origins | Recommended |
| `BIOSUITE_DEV_MODE` | Set to `1` to expose `/docs` and `/openapi.json` unauthenticated | Development only |

Copy [`.env.example`](../.env.example) as a starting point. Never commit real credentials;
`.env` is git-ignored while `.env.example` is tracked deliberately.

### Rotating credentials

Credentials that were committed to this repository's history in earlier revisions must be
treated as **compromised**. Rotate the API key, the JWT signing secret and the admin password
before deploying, and rotate any value that has ever appeared in a commit, a log or a CI
configuration. Removing a secret from the working tree does not remove it from history.

## Security testing

The CI pipeline runs a dedicated security-regression suite covering authentication bypass,
JWT forgery, default credentials, path traversal and arbitrary file read, CORS reflection and
unsafe CLI evaluation, alongside CodeQL static analysis for both the Python code and the
GitHub Actions workflows. These are engineering controls, not a guarantee: the project has not
had an external security audit and is not validated for regulated or clinical data.
