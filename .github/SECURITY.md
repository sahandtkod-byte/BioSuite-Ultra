# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 4.2.x   | ✅ Active support  |
| 4.1.x   | ⚠️ Security only   |
| < 4.1   | ❌ End of life     |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report
1. Email: **sahandtkod@gmail.com**
2. Subject: `[SECURITY] BioSuite Ultra - <brief description>`
3. Include: affected version, description, steps to reproduce, impact assessment

### What to Expect
- **Acknowledgment**: within 48 hours
- **Assessment**: within 7 days
- **Fix timeline**: Critical (1 week), High (2 weeks), Medium (1 month)
- **Credit**: reporters will be credited in release notes (unless anonymous preferred)

### Scope
- REST API authentication & authorization
- CLI input handling (injection, eval)
- File I/O (path traversal, arbitrary read/write)
- Dependency vulnerabilities
- Docker/container escape vectors

### Out of Scope
- Social engineering attacks
- DoS against self-hosted instances
- Issues in third-party dependencies (report upstream)
