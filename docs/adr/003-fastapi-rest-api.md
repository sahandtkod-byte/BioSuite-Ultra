# ADR-003: Use FastAPI for REST API

## Status
Accepted

## Context
BioSuite exposes analysis via REST API. Options: Flask, Django REST, FastAPI.

## Decision
Use **FastAPI** with Pydantic models.

## Rationale
- Async-native — non-blocking for long-running analyses
- Auto-generated OpenAPI/Swagger docs at `/docs`
- Pydantic request/response validation
- Dependency injection for auth, rate limiting
- Performance on par with Go/Rust frameworks (uvicorn + uvloop)

## Consequences
- Requires Python 3.10+ for full feature set
- Async code throughout API layer (no blocking in request handlers)
- uvicorn used as ASGI server
