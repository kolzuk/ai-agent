"""Health check endpoints."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from ..config import settings
from ..database import db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Enhanced health check endpoint with SDLC system monitoring."""
    try:
        health_status = {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {}
        }
        
        # Test database connection
        try:
            with db_manager.get_session() as db:
                db.execute(text("SELECT 1"))
                health_status["components"]["database"] = "healthy"
        except Exception as db_e:
            health_status["components"]["database"] = f"unhealthy: {str(db_e)}"
            health_status["status"] = "degraded"
        
        # Check active iterations health
        try:
            with db_manager.get_session() as db:
                from sqlalchemy import func, and_
                from ..database import IssueIteration
                
                # Count stuck iterations (waiting for CI too long)
                stuck_iterations = db.query(func.count(IssueIteration.id)).filter(
                    and_(
                        IssueIteration.is_active == True,
                        IssueIteration.status == "waiting_ci",
                        IssueIteration.updated_at < datetime.utcnow() - timedelta(minutes=10)
                    )
                ).scalar()
                
                # Count locked iterations
                locked_iterations = db.query(func.count(IssueIteration.id)).filter(
                    and_(
                        IssueIteration.is_active == True,
                        IssueIteration.is_locked == True
                    )
                ).scalar()
                
                # Count total active iterations
                active_iterations = db.query(func.count(IssueIteration.id)).filter(
                    IssueIteration.is_active == True
                ).scalar()
                
                health_status["components"]["iterations"] = {
                    "status": "healthy",
                    "active_count": active_iterations,
                    "stuck_count": stuck_iterations,
                    "locked_count": locked_iterations
                }
                
                # Mark as degraded if too many stuck iterations
                if stuck_iterations > 5:
                    health_status["components"]["iterations"]["status"] = "degraded"
                    health_status["status"] = "degraded"
                    
        except Exception as iter_e:
            health_status["components"]["iterations"] = f"unhealthy: {str(iter_e)}"
            health_status["status"] = "degraded"
        
        # Determine overall status code
        status_code = 200 if health_status["status"] == "ok" else 503
        
        return JSONResponse(content=health_status, status_code=status_code)
        
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