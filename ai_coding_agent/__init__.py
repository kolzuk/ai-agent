"""AI Coding Agent - Automated GitHub SDLC system."""

__version__ = "0.1.0"
__author__ = "AI Coding Agent"
__email__ = "agent@example.com"

# Note: Config is not imported here to avoid validation errors in FastAPI service
# CodeAgent and ReviewerAgent are imported directly when needed with proper parameters

__all__ = ["CodeAgent", "ReviewerAgent"]