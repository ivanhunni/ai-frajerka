# api/__init__.py
"""
API Module for AI-Frajerka.
Provides REST and WebSocket endpoints using FastAPI.
"""
from .app import app

__all__ = ["app"]