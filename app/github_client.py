"""GitHub API client for GitHub App integration."""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import httpx

from .github_app.auth import github_app_auth
from .config import settings

logger = logging.getLogger(__name__)


class GitHubAppClient:
    """GitHub API client using GitHub App authentication."""
    
    def __init__(self, installation_id: int):
        self.installation_id = installation_id
        self._client: Optional[httpx.AsyncClient] = None
        self._file_cache = {}  # Кэш для файловых структур
        self._cache_ttl = timedelta(minutes=5)  # TTL для кэша
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = await github_app_auth.get_authenticated_client(self.installation_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def get_issue(self, owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
        """Get issue details."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Get pull request details."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def create_branch(self, owner: str, repo: str, branch_name: str, base_sha: str) -> Dict[str, Any]:
        """Create a new branch."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
        
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }
        
        response = await self._client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def update_branch(self, owner: str, repo: str, branch_name: str, new_sha: str) -> Dict[str, Any]:
        """Update an existing branch to point to a new commit."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
        
        data = {
            "sha": new_sha,
            "force": True  # Force update even if it's not a fast-forward
        }
        
        response = await self._client.patch(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Get repository default branch."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        
        response = await self._client.get(url)
        response.raise_for_status()
        repo_data = response.json()
        return repo_data["default_branch"]
    
    async def get_branch_sha(self, owner: str, repo: str, branch: str) -> str:
        """Get SHA of a branch."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"
        
        response = await self._client.get(url)
        response.raise_for_status()
        ref_data = response.json()
        return ref_data["object"]["sha"]
    
    async def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a file in the repository."""
        import base64
        
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        # Encode content to base64
        encoded_content = base64.b64encode(content.encode()).decode()
        
        data = {
            "message": message,
            "content": encoded_content,
            "branch": branch
        }
        
        if sha:
            data["sha"] = sha
        
        response = await self._client.put(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get file content from repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        params = {}
        if branch:
            params["ref"] = branch
        
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str
    ) -> Dict[str, Any]:
        """Create a pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        response = await self._client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        head: Optional[str] = None,
        base: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List pull requests."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        
        params = {"state": state}
        if head:
            params["head"] = head
        if base:
            params["base"] = base
        
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def update_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        
        response = await self._client.patch(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def close_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> Dict[str, Any]:
        """Close a pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        
        data = {"state": "closed"}
        
        response = await self._client.patch(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def create_issue_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str
    ) -> Dict[str, Any]:
        """Create a comment on an issue or pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        
        data = {"body": body}
        
        response = await self._client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT"  # APPROVE, REQUEST_CHANGES, COMMENT
    ) -> Dict[str, Any]:
        """Create a pull request review."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        
        data = {
            "body": body,
            "event": event
        }
        
        response = await self._client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def get_pull_request_files(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> List[Dict[str, Any]]:
        """Get files changed in a pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_workflow_runs(
        self,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get workflow runs for a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
        
        params = {}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status
        
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()["workflow_runs"]
    
    async def get_workflow_run(
        self,
        owner: str,
        repo: str,
        run_id: int
    ) -> Dict[str, Any]:
        """Get specific workflow run."""
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_workflow_run_jobs(
        self,
        owner: str,
        repo: str,
        run_id: int
    ) -> List[Dict[str, Any]]:
        """Get jobs for a workflow run."""
        url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()["jobs"]
    
    async def get_commit_status(
        self,
        owner: str,
        repo: str,
        sha: str
    ) -> Dict[str, Any]:
        """Get commit status."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/status"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def get_check_runs(
        self,
        owner: str,
        repo: str,
        sha: str
    ) -> List[Dict[str, Any]]:
        """Get check runs for a commit."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/check-runs"
        
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()["check_runs"]
    
    async def list_repository_files_recursive(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: str = None,
        max_depth: int = 10,
        max_files: int = 1000,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Recursively list all files in repository with caching and limits."""
        
        # Проверить кэш
        cache_key = f"{owner}/{repo}:{branch or 'main'}:{path}"
        if use_cache and cache_key in self._file_cache:
            cached_data, cached_time = self._file_cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.info(f"Using cached file list for {cache_key}")
                return cached_data
        
        all_files = []
        
        async def _traverse_directory(current_path: str, depth: int = 0):
            if depth > max_depth:
                logger.warning(f"Max depth {max_depth} reached for path {current_path}")
                return
            
            if len(all_files) >= max_files:
                logger.warning(f"Max files limit {max_files} reached")
                return
                
            try:
                url = f"https://api.github.com/repos/{owner}/{repo}/contents/{current_path}"
                params = {}
                if branch:
                    params["ref"] = branch
                
                response = await self._client.get(url, params=params)
                response.raise_for_status()
                contents = response.json()
                
                if isinstance(contents, dict):
                    # Single file
                    all_files.append(contents)
                else:
                    # Directory listing
                    for item in contents:
                        if len(all_files) >= max_files:
                            break
                            
                        if item["type"] == "file":
                            all_files.append(item)
                        elif item["type"] == "dir":
                            # Пропустить некоторые директории для производительности
                            if item["name"] in [".git", "node_modules", ".idea", "target", "build"]:
                                logger.debug(f"Skipping directory {item['path']}")
                                continue
                            await _traverse_directory(item["path"], depth + 1)
                            
            except Exception as e:
                logger.error(f"Failed to traverse {current_path}: {e}")
        
        await _traverse_directory(path)
        
        # Кэшировать результат
        if use_cache:
            self._file_cache[cache_key] = (all_files, datetime.utcnow())
        
        logger.info(f"Recursively found {len(all_files)} files in {owner}/{repo}")
        return all_files

    async def list_repository_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        branch: str = None
    ) -> List[Dict[str, Any]]:
        """List files in repository directory (now with recursive support)."""
        return await self.list_repository_files_recursive(owner, repo, path, branch)
    
    async def get_branch_protection(
        self,
        owner: str,
        repo: str,
        branch: str
    ) -> Optional[Dict[str, Any]]:
        """Get branch protection rules."""
        url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}/protection"
        
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"No branch protection found for {owner}/{repo}:{branch}")
                return None
            logger.error(f"Failed to get branch protection: {e}")
            return None
    
    async def get_required_status_checks(
        self,
        owner: str,
        repo: str,
        branch: str
    ) -> List[str]:
        """Get required status checks from branch protection."""
        try:
            protection = await self.get_branch_protection(owner, repo, branch)
            if not protection:
                logger.info(f"No branch protection for {owner}/{repo}:{branch}, treating all checks as optional")
                return []
            
            required_checks = []
            
            # Get required status checks
            status_checks = protection.get("required_status_checks", {})
            if status_checks:
                # Get contexts (legacy status checks)
                contexts = status_checks.get("contexts", [])
                required_checks.extend(contexts)
                
                # Get checks (newer check runs)
                checks = status_checks.get("checks", [])
                for check in checks:
                    if isinstance(check, dict):
                        required_checks.append(check.get("context", ""))
                    else:
                        required_checks.append(str(check))
            
            logger.info(f"Required checks for {owner}/{repo}:{branch}: {required_checks}")
            return [check for check in required_checks if check]  # Filter out empty strings
            
        except Exception as e:
            logger.error(f"Failed to get required status checks: {e}")
            # Fallback: treat all checks as required for safety
            return []
    
    async def get_pr_ci_status(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        expected_sha: str = None
    ) -> Dict[str, Any]:
        """Get comprehensive CI status for a PR with required checks logic."""
        try:
            # Get PR details to get the head SHA and base branch
            pr_data = await self.get_pull_request(owner, repo, pr_number)
            head_sha = pr_data["head"]["sha"]
            base_branch = pr_data["base"]["ref"]
            
            # Validate SHA if provided
            if expected_sha and head_sha != expected_sha:
                logger.warning(f"PR head SHA {head_sha[:8]} doesn't match expected {expected_sha[:8]}")
                return {
                    "overall_conclusion": "unknown",
                    "overall_status": "unknown",
                    "error": f"SHA mismatch: expected {expected_sha[:8]}, got {head_sha[:8]}",
                    "head_sha": head_sha
                }
            
            logger.info(f"Checking CI status for PR #{pr_number} at SHA {head_sha[:8]}")
            
            # Get required checks from branch protection
            required_checks = await self.get_required_status_checks(owner, repo, base_branch)
            
            # Get check runs and commit status
            check_runs = await self.get_check_runs(owner, repo, head_sha)
            commit_status = await self.get_commit_status(owner, repo, head_sha)
            
            # Filter to only required checks
            required_check_runs = []
            required_statuses = []
            
            if required_checks:
                # Filter check runs to only required ones
                for check in check_runs:
                    if check["name"] in required_checks:
                        required_check_runs.append(check)
                
                # Filter commit statuses to only required ones
                for status in commit_status.get("statuses", []):
                    if status["context"] in required_checks:
                        required_statuses.append(status)
                
                logger.info(f"Found {len(required_check_runs)} required check runs and {len(required_statuses)} required statuses out of {len(check_runs)} total checks")
            else:
                # No branch protection - treat all checks as required for safety
                required_check_runs = check_runs
                required_statuses = commit_status.get("statuses", [])
                logger.info(f"No branch protection found, treating all {len(check_runs)} checks as required")
            
            # Analyze only required checks
            overall_conclusion = "success"
            failed_checks = []
            pending_checks = []
            
            # Required check runs analysis
            for check in required_check_runs:
                check_name = check["name"]
                if check["conclusion"] in ["failure", "cancelled", "timed_out"]:
                    overall_conclusion = "failure"
                    failed_checks.append(check_name)
                    logger.info(f"Required check '{check_name}' failed: {check['conclusion']}")
                elif check["conclusion"] in ["neutral", "skipped"]:
                    logger.info(f"Required check '{check_name}' skipped/neutral, ignoring")
                    continue
                elif check["status"] in ["queued", "in_progress"] or not check["conclusion"]:
                    overall_conclusion = "pending"
                    pending_checks.append(check_name)
                    logger.info(f"Required check '{check_name}' still pending: {check['status']}")
                elif check["conclusion"] == "success":
                    logger.info(f"Required check '{check_name}' passed")
            
            # Required commit status analysis
            for status in required_statuses:
                context = status["context"]
                if status["state"] in ["failure", "error"]:
                    overall_conclusion = "failure"
                    failed_checks.append(context)
                    logger.info(f"Required status '{context}' failed: {status['state']}")
                elif status["state"] == "pending":
                    overall_conclusion = "pending"
                    pending_checks.append(context)
                    logger.info(f"Required status '{context}' still pending")
                elif status["state"] == "success":
                    logger.info(f"Required status '{context}' passed")
            
            # If no required checks found and no checks exist, consider it success
            if not required_check_runs and not required_statuses and not check_runs:
                logger.info("No checks found, considering CI as success")
                overall_conclusion = "success"
            
            logger.info(f"Overall CI conclusion for PR #{pr_number}: {overall_conclusion} (based on {len(required_check_runs + required_statuses)} required checks)")
            
            return {
                "overall_conclusion": overall_conclusion,
                "overall_status": "completed" if overall_conclusion in ["success", "failure"] else "pending",
                "check_runs": check_runs,
                "required_check_runs": required_check_runs,
                "commit_status": commit_status,
                "required_statuses": required_statuses,
                "required_checks": required_checks,
                "failed_checks": failed_checks,
                "pending_checks": pending_checks,
                "head_sha": head_sha
            }
            
        except Exception as e:
            logger.error(f"Failed to get CI status for PR #{pr_number}: {e}")
            return {
                "overall_conclusion": "unknown",
                "overall_status": "unknown",
                "error": str(e),
                "head_sha": expected_sha or "unknown"
            }


def determine_ci_conclusion_from_db_and_api(
    iteration_data: dict,
    api_ci_details: dict = None,
    current_head_sha: str = None
) -> tuple[str, str]:
    """
    Determine CI conclusion using DB-first logic with API fallback.
    
    Args:
        iteration_data: Dictionary with iteration data (pr_head_sha, ci_status_source, etc.)
        api_ci_details: Optional API CI details
        current_head_sha: Current PR head SHA
    
    Returns:
        tuple: (ci_conclusion, data_source)
    """
    from datetime import datetime, timedelta
    
    # Check if we have recent webhook data for the current SHA
    if (iteration_data.get('pr_head_sha') and current_head_sha and
        iteration_data['pr_head_sha'] == current_head_sha and
        iteration_data.get('ci_status_source') == "webhook"):
        
        if (iteration_data.get('ci_status_updated_at') and
            datetime.utcnow() - iteration_data['ci_status_updated_at'] < timedelta(minutes=5)):
            return iteration_data.get('last_ci_conclusion', 'unknown'), "webhook (recent)"
    
    # Fallback to API data
    if api_ci_details and api_ci_details.get("overall_conclusion") != "unknown":
        return api_ci_details["overall_conclusion"], "api"
    
    # Final fallback to stored data
    return iteration_data.get('last_ci_conclusion', 'unknown'), "stored"


async def get_github_client(installation_id: int) -> GitHubAppClient:
    """Get authenticated GitHub client for installation."""
    return GitHubAppClient(installation_id)