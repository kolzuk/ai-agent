"""Health check endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from ..config import settings
from ..database import db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        with db_manager.get_session() as db:
            db.execute(text("SELECT 1"))
        
        return JSONResponse(
            content={
                "status": "ok",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "database": "connected"
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            },
            status_code=503
        )


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check database
        with db_manager.get_session() as db:
            db.execute(text("SELECT 1"))
        
        # Check configuration
        required_config = [
            settings.github_app_id,
            settings.github_app_private_key,
            settings.github_webhook_secret
        ]
        
        if settings.use_yandex_gpt:
            required_config.extend([
                settings.yandex_gpt_api_key,
                settings.yandex_gpt_folder_id
            ])
        else:
            required_config.append(settings.openai_api_key)
        
        if not all(required_config):
            raise ValueError("Missing required configuration")
        
        return JSONResponse(
            content={
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "connected",
                "configuration": "valid"
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            },
            status_code=503
        )