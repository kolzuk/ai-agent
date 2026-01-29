"""Webhook handlers for GitHub events."""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..config import settings
from ..orchestrator import orchestrator
from ..github_app.auth import github_app_auth

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle GitHub webhook events."""
    try:
        # Log all headers for debugging
        logger.info(f"Webhook headers: {dict(request.headers)}")
        
        # Get headers
        event_type = request.headers.get("X-GitHub-Event")
        delivery_id = request.headers.get("X-GitHub-Delivery")
        
        if not event_type:
            logger.error("Missing X-GitHub-Event header")
            raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")
                
        # Get payload
        payload = await request.body()
        logger.info(f"Webhook payload size: {len(payload)} bytes")
        
        # Skip signature verification for now
        logger.info("Signature verification disabled for testing")
        
        # Parse payload
        try:
            data = json.loads(payload.decode())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON payload: {e}")
            logger.debug(f"Payload content: {payload.decode()[:500]}...")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        logger.info(f"Received webhook: {event_type} (delivery: {delivery_id})")
        
        # Route to appropriate handler
        if event_type == "issues":
            background_tasks.add_task(handle_issues_event, data)
        elif event_type == "pull_request":
            background_tasks.add_task(handle_pull_request_event, data)
        elif event_type == "check_suite":
            background_tasks.add_task(handle_check_suite_event, data)
        elif event_type == "ping":
            logger.info("Received ping event")
        else:
            logger.info(f"Unhandled event type: {event_type}")
        
        return JSONResponse(content={"status": "ok"}, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def handle_issues_event(data: Dict[str, Any]) -> None:
    """Handle issues webhook events."""
    try:
        action = data.get("action")
        issue = data.get("issue", {})
        repository = data.get("repository", {})
        installation = data.get("installation", {})
        
        if action not in ["opened", "reopened"]:
            logger.info(f"Ignoring issues action: {action}")
            return
        
        repo_full_name = repository.get("full_name")
        issue_number = issue.get("number")
        installation_id = installation.get("id")
        
        if not all([repo_full_name, issue_number, installation_id]):
            logger.error("Missing required fields in issues event")
            return
        
        logger.info(f"Starting SDLC cycle for issue {repo_full_name}#{issue_number}")
        
        # Check if there's already a failed iteration for this issue
        from ..database import db_manager
        existing_iteration = db_manager.get_active_iteration(repo_full_name, issue_number)
        
        if existing_iteration:
            logger.info(f"Active iteration exists for issue #{issue_number}, using existing")
            iteration = existing_iteration
        else:
            # Check if there are any completed/failed iterations for this issue
            # If so, restart instead of start
            with db_manager.get_session() as db:
                from sqlalchemy import and_
                from ..database import IssueIteration
                previous_iterations = db.query(IssueIteration).filter(
                    and_(
                        IssueIteration.repo_full_name == repo_full_name,
                        IssueIteration.issue_number == issue_number,
                        IssueIteration.is_active == False
                    )
                ).count()
            
            if previous_iterations > 0:
                logger.info(f"Found {previous_iterations} previous iterations, restarting cycle")
                iteration = await orchestrator.restart_issue_cycle(
                    repo_full_name=repo_full_name,
                    issue_number=issue_number,
                    installation_id=installation_id
                )
            else:
                logger.info(f"No previous iterations found, starting new cycle")
                iteration = await orchestrator.start_issue_cycle(
                    repo_full_name=repo_full_name,
                    issue_number=issue_number,
                    installation_id=installation_id
                )
        
        if iteration:
            logger.info(f"Successfully started SDLC cycle (iteration ID: {iteration.id})")
        else:
            logger.error("Failed to start SDLC cycle")
        
    except Exception as e:
        logger.error(f"Error handling issues event: {e}", exc_info=True)


async def handle_pull_request_event(data: Dict[str, Any]) -> None:
    """Handle pull request webhook events."""
    try:
        action = data.get("action")
        pull_request = data.get("pull_request", {})
        repository = data.get("repository", {})
        installation = data.get("installation", {})
        
        if action not in ["opened", "synchronize", "reopened"]:
            logger.info(f"Ignoring pull_request action: {action}")
            return
        
        repo_full_name = repository.get("full_name")
        pr_number = pull_request.get("number")
        installation_id = installation.get("id")
        
        if not all([repo_full_name, pr_number, installation_id]):
            logger.error("Missing required fields in pull_request event")
            return
        
        # Check if this PR is managed by our system
        from ..database import db_manager
        iteration = db_manager.get_iteration_by_pr(repo_full_name, pr_number)
        
        if not iteration:
            logger.info(f"PR {repo_full_name}#{pr_number} is not managed by AI Coding Agent")
            return
        
        logger.info(f"PR {repo_full_name}#{pr_number} updated, checking if review needed")
        
        # For synchronize events (new commits), we might want to wait for CI
        if action == "synchronize":
            # Update iteration status to wait for CI
            db_manager.update_iteration(
                iteration.id,
                status="waiting_ci"
            )
            logger.info(f"Updated iteration {iteration.id} to wait for CI")
        
    except Exception as e:
        logger.error(f"Error handling pull_request event: {e}", exc_info=True)


async def handle_check_suite_event(data: Dict[str, Any]) -> None:
    """Handle check suite webhook events."""
    try:
        action = data.get("action")
        check_suite = data.get("check_suite", {})
        repository = data.get("repository", {})
        installation = data.get("installation", {})
        
        if action != "completed":
            logger.info(f"Ignoring check_suite action: {action}")
            return
        
        repo_full_name = repository.get("full_name")
        head_branch = check_suite.get("head_branch")
        head_sha = check_suite.get("head_sha")  # Extract head SHA
        status = check_suite.get("status")
        conclusion = check_suite.get("conclusion")
        installation_id = installation.get("id")
        
        if not all([repo_full_name, head_branch, head_sha, status, installation_id]):
            logger.error("Missing required fields in check_suite event")
            return
        
        logger.info(f"Check suite completed for {repo_full_name}:{head_branch}@{head_sha[:8]} - {status}/{conclusion}")
        
        # Find associated PRs
        pull_requests = check_suite.get("pull_requests", [])
        if not pull_requests:
            logger.info("No pull requests associated with this check suite")
            return
        
        # Handle each associated PR with SHA binding
        for pr_info in pull_requests:
            pr_number = pr_info.get("number")
            if pr_number:
                logger.info(f"Processing CI completion for PR #{pr_number} with SHA {head_sha[:8]}")
                await orchestrator.handle_ci_completion(
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    ci_status=status,
                    ci_conclusion=conclusion,
                    head_sha=head_sha
                )
        
    except Exception as e:
        logger.error(f"Error handling check_suite event: {e}", exc_info=True)