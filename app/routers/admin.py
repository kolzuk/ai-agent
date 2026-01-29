"""Admin API endpoints for manual control."""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..database import db_manager, IssueIteration, IterationStatus
from ..orchestrator import orchestrator
from ..github_app.auth import github_app_auth

logger = logging.getLogger(__name__)

router = APIRouter()


class IterationResponse(BaseModel):
    """Response model for iteration data."""
    id: int
    repo_full_name: str
    issue_number: int
    pr_number: Optional[int]
    current_iteration: int
    max_iterations: int
    status: str
    issue_title: Optional[str]
    branch_name: Optional[str]
    last_review_score: Optional[int]
    last_review_recommendation: Optional[str]
    last_ci_status: Optional[str]
    last_ci_conclusion: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]


class StartIssueRequest(BaseModel):
    """Request model for starting issue processing."""
    max_iterations: Optional[int] = None


class ReviewPRRequest(BaseModel):
    """Request model for manual PR review."""
    force: bool = False


@router.post("/run/issue/{owner}/{repo}/{issue_number}")
async def start_issue_manually(
    owner: str,
    repo: str,
    issue_number: int,
    request: StartIssueRequest,
    background_tasks: BackgroundTasks
):
    """Manually start SDLC cycle for an issue."""
    try:
        repo_full_name = f"{owner}/{repo}"
        
        # Get installation ID
        installation_id = await github_app_auth.get_installation_id(owner, repo)
        if not installation_id:
            raise HTTPException(
                status_code=404,
                detail=f"GitHub App not installed on {repo_full_name}"
            )
        
        # Check if there's already an active iteration
        existing = db_manager.get_active_iteration(repo_full_name, issue_number)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Active iteration already exists for issue #{issue_number}"
            )
        
        logger.info(f"Manually starting SDLC cycle for {repo_full_name}#{issue_number}")
        
        # Start SDLC cycle in background
        background_tasks.add_task(
            _start_issue_background,
            repo_full_name,
            issue_number,
            installation_id,
            request.max_iterations
        )
        
        return JSONResponse(
            content={
                "status": "started",
                "message": f"SDLC cycle started for issue #{issue_number}",
                "repo": repo_full_name,
                "issue_number": issue_number
            },
            status_code=202
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start issue manually: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart/issue/{owner}/{repo}/{issue_number}")
async def restart_issue_manually(
    owner: str,
    repo: str,
    issue_number: int,
    request: StartIssueRequest,
    background_tasks: BackgroundTasks
):
    """Restart SDLC cycle for an issue (even if previous iterations failed)."""
    try:
        repo_full_name = f"{owner}/{repo}"
        
        # Get installation ID
        installation_id = await github_app_auth.get_installation_id(owner, repo)
        if not installation_id:
            raise HTTPException(
                status_code=404,
                detail=f"GitHub App not installed on {repo_full_name}"
            )
        
        logger.info(f"Manually restarting SDLC cycle for {repo_full_name}#{issue_number}")
        
        # Restart SDLC cycle in background
        background_tasks.add_task(
            _restart_issue_background,
            repo_full_name,
            issue_number,
            installation_id,
            request.max_iterations
        )
        
        return JSONResponse(
            content={
                "status": "restarted",
                "message": f"SDLC cycle restarted for issue #{issue_number}",
                "repo": repo_full_name,
                "issue_number": issue_number
            },
            status_code=202
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart issue manually: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _restart_issue_background(
    repo_full_name: str,
    issue_number: int,
    installation_id: int,
    max_iterations: Optional[int]
):
    """Background task to restart issue processing."""
    try:
        # Use the restart method from orchestrator
        await orchestrator.restart_issue_cycle(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            installation_id=installation_id
        )
    except Exception as e:
        logger.error(f"Background issue restart failed: {e}", exc_info=True)


async def _start_issue_background(
    repo_full_name: str,
    issue_number: int,
    installation_id: int,
    max_iterations: Optional[int]
):
    """Background task to start issue processing."""
    try:
        # Create iteration with custom max_iterations if provided
        if max_iterations:
            # We need to create the iteration manually to set custom max_iterations
            from ..github_client import get_github_client
            
            owner, repo = repo_full_name.split("/")
            async with await get_github_client(installation_id) as github:
                issue_data = await github.get_issue(owner, repo, issue_number)
            
            iteration = db_manager.create_iteration(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                installation_id=installation_id,
                issue_title=issue_data.get("title"),
                issue_body=issue_data.get("body"),
                max_iterations=max_iterations
            )
            
            # Start the cycle manually
            await orchestrator._run_code_iteration(iteration)
        else:
            # Use default flow
            await orchestrator.start_issue_cycle(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                installation_id=installation_id
            )
    except Exception as e:
        logger.error(f"Background issue processing failed: {e}", exc_info=True)


@router.post("/run/review/{owner}/{repo}/{pr_number}")
async def review_pr_manually(
    owner: str,
    repo: str,
    pr_number: int,
    request: ReviewPRRequest,
    background_tasks: BackgroundTasks
):
    """Manually trigger PR review."""
    try:
        repo_full_name = f"{owner}/{repo}"
        
        # Find iteration by PR
        iteration = db_manager.get_iteration_by_pr(repo_full_name, pr_number)
        if not iteration and not request.force:
            raise HTTPException(
                status_code=404,
                detail=f"No active iteration found for PR #{pr_number}. Use force=true to review anyway."
            )
        
        if not iteration and request.force:
            # Get installation ID for forced review
            installation_id = await github_app_auth.get_installation_id(owner, repo)
            if not installation_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"GitHub App not installed on {repo_full_name}"
                )
            
            # Create a temporary iteration for forced review
            iteration = db_manager.create_iteration(
                repo_full_name=repo_full_name,
                issue_number=0,  # Dummy issue number for forced review
                installation_id=installation_id,
                issue_title="Manual PR Review",
                issue_body="Manually triggered PR review",
                max_iterations=1
            )
            
            # Update with PR number
            db_manager.update_iteration(iteration.id, pr_number=pr_number)
        
        logger.info(f"Manually triggering review for {repo_full_name} PR#{pr_number}")
        
        # Trigger review in background
        background_tasks.add_task(_review_pr_background, iteration)
        
        return JSONResponse(
            content={
                "status": "started",
                "message": f"Review started for PR #{pr_number}",
                "repo": repo_full_name,
                "pr_number": pr_number,
                "iteration_id": iteration.id
            },
            status_code=202
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start PR review manually: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _review_pr_background(iteration: IssueIteration):
    """Background task to review PR."""
    try:
        # Update status and trigger review
        db_manager.update_iteration(
            iteration.id,
            status=IterationStatus.REVIEWING,
            last_ci_status="unknown",
            last_ci_conclusion="unknown"
        )
        
        await orchestrator._run_review_iteration(iteration)
    except Exception as e:
        logger.error(f"Background PR review failed: {e}", exc_info=True)


@router.get("/status/{owner}/{repo}/{issue_number}")
async def get_issue_status(owner: str, repo: str, issue_number: int):
    """Get status of an issue's SDLC cycle."""
    try:
        repo_full_name = f"{owner}/{repo}"
        
        iteration = db_manager.get_active_iteration(repo_full_name, issue_number)
        if not iteration:
            raise HTTPException(
                status_code=404,
                detail=f"No active iteration found for issue #{issue_number}"
            )
        
        return IterationResponse(
            id=iteration.id,
            repo_full_name=iteration.repo_full_name,
            issue_number=iteration.issue_number,
            pr_number=iteration.pr_number,
            current_iteration=iteration.current_iteration,
            max_iterations=iteration.max_iterations,
            status=iteration.status,
            issue_title=iteration.issue_title,
            branch_name=iteration.branch_name,
            last_review_score=iteration.last_review_score,
            last_review_recommendation=iteration.last_review_recommendation,
            last_ci_status=iteration.last_ci_status,
            last_ci_conclusion=iteration.last_ci_conclusion,
            created_at=iteration.created_at.isoformat(),
            updated_at=iteration.updated_at.isoformat(),
            completed_at=iteration.completed_at.isoformat() if iteration.completed_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get issue status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/iterations")
async def list_active_iterations():
    """List all active iterations."""
    try:
        iterations = db_manager.get_all_active_iterations()
        
        return [
            IterationResponse(
                id=iteration.id,
                repo_full_name=iteration.repo_full_name,
                issue_number=iteration.issue_number,
                pr_number=iteration.pr_number,
                current_iteration=iteration.current_iteration,
                max_iterations=iteration.max_iterations,
                status=iteration.status,
                issue_title=iteration.issue_title,
                branch_name=iteration.branch_name,
                last_review_score=iteration.last_review_score,
                last_review_recommendation=iteration.last_review_recommendation,
                last_ci_status=iteration.last_ci_status,
                last_ci_conclusion=iteration.last_ci_conclusion,
                created_at=iteration.created_at.isoformat(),
                updated_at=iteration.updated_at.isoformat(),
                completed_at=iteration.completed_at.isoformat() if iteration.completed_at else None
            )
            for iteration in iterations
        ]
        
    except Exception as e:
        logger.error(f"Failed to list iterations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/iterations/{iteration_id}/cancel")
async def cancel_iteration(iteration_id: int):
    """Cancel an active iteration."""
    try:
        iteration = db_manager.update_iteration(
            iteration_id,
            status=IterationStatus.CANCELLED,
            is_active=False
        )
        
        if not iteration:
            raise HTTPException(
                status_code=404,
                detail=f"Iteration {iteration_id} not found"
            )
        
        logger.info(f"Cancelled iteration {iteration_id}")
        
        return JSONResponse(
            content={
                "status": "cancelled",
                "message": f"Iteration {iteration_id} has been cancelled",
                "iteration_id": iteration_id
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel iteration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        with db_manager.get_session() as db:
            # Count iterations by status
            from sqlalchemy import func
            
            stats = db.query(
                IssueIteration.status,
                func.count(IssueIteration.id).label('count')
            ).group_by(IssueIteration.status).all()
            
            status_counts = {status: count for status, count in stats}
            
            # Get total iterations
            total_iterations = db.query(func.count(IssueIteration.id)).scalar()
            
            # Get active iterations
            active_iterations = db.query(func.count(IssueIteration.id)).filter(
                IssueIteration.is_active == True
            ).scalar()
            
            return {
                "total_iterations": total_iterations,
                "active_iterations": active_iterations,
                "status_breakdown": status_counts,
                "timestamp": db.query(func.now()).scalar().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))