"""GitHub client for interacting with GitHub API."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import git
from github import Github
from github.Issue import Issue
from github.PullRequest import PullRequest
from github.Repository import Repository

# Configuration will be passed as parameters instead of importing global config

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API and Git operations."""

    def __init__(self, github_token: str, repo_owner: str, repo_name: str) -> None:
        """Initialize the GitHub client."""
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github = Github(self.github_token)
        self.repo = self.github.get_repo(f"{self.repo_owner}/{self.repo_name}")

    def get_issue(self, issue_number: int) -> Issue:
        """Get an issue by number."""
        try:
            return self.repo.get_issue(issue_number)
        except Exception as e:
            logger.error(f"Error getting issue {issue_number}: {e}")
            raise

    def get_pull_request(self, pr_number: int) -> PullRequest:
        """Get a pull request by number."""
        try:
            return self.repo.get_pull(pr_number)
        except Exception as e:
            logger.error(f"Error getting pull request {pr_number}: {e}")
            raise

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> PullRequest:
        """Create a new pull request."""
        try:
            pr = self.repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )
            logger.info(f"Created pull request #{pr.number}: {title}")
            return pr
        except Exception as e:
            logger.error(f"Error creating pull request: {e}")
            raise

    def add_comment_to_pr(self, pr_number: int, comment: str) -> None:
        """Add a comment to a pull request."""
        try:
            pr = self.get_pull_request(pr_number)
            pr.create_issue_comment(comment)
            logger.info(f"Added comment to PR #{pr_number}")
        except Exception as e:
            logger.error(f"Error adding comment to PR {pr_number}: {e}")
            raise

    def get_pr_files(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get files changed in a pull request."""
        try:
            pr = self.get_pull_request(pr_number)
            files = []
            for file in pr.get_files():
                files.append({
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch,
                    "raw_url": file.raw_url,
                })
            return files
        except Exception as e:
            logger.error(f"Error getting PR files for {pr_number}: {e}")
            raise

    def get_pr_diff(self, pr_number: int) -> str:
        """Get the diff for a pull request."""
        try:
            pr = self.get_pull_request(pr_number)
            return pr.diff_url
        except Exception as e:
            logger.error(f"Error getting PR diff for {pr_number}: {e}")
            raise

    def close_issue(self, issue_number: int, comment: Optional[str] = None) -> None:
        """Close an issue with optional comment."""
        try:
            issue = self.get_issue(issue_number)
            if comment:
                issue.create_comment(comment)
            issue.edit(state="closed")
            logger.info(f"Closed issue #{issue_number}")
        except Exception as e:
            logger.error(f"Error closing issue {issue_number}: {e}")
            raise

    def create_branch(self, branch_name: str, base_branch: str = "main") -> None:
        """Create a new branch from base branch."""
        try:
            base_ref = self.repo.get_git_ref(f"heads/{base_branch}")
            self.repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_ref.object.sha
            )
            logger.info(f"Created branch {branch_name} from {base_branch}")
        except Exception as e:
            logger.error(f"Error creating branch {branch_name}: {e}")
            raise

    def update_file(
        self,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str,
    ) -> None:
        """Update or create a file in the repository."""
        try:
            # Try to get existing file
            try:
                file = self.repo.get_contents(file_path, ref=branch)
                # Update existing file
                self.repo.update_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    sha=file.sha,
                    branch=branch,
                )
                logger.info(f"Updated file {file_path} in branch {branch}")
            except Exception:
                # Create new file
                self.repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    branch=branch,
                )
                logger.info(f"Created file {file_path} in branch {branch}")
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")
            raise

    def delete_file(
        self,
        file_path: str,
        commit_message: str,
        branch: str,
    ) -> None:
        """Delete a file from the repository."""
        try:
            file = self.repo.get_contents(file_path, ref=branch)
            self.repo.delete_file(
                path=file_path,
                message=commit_message,
                sha=file.sha,
                branch=branch,
            )
            logger.info(f"Deleted file {file_path} from branch {branch}")
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise

    def get_file_content(self, file_path: str, branch: str = "main") -> str:
        """Get content of a file from the repository."""
        try:
            file = self.repo.get_contents(file_path, ref=branch)
            return file.decoded_content.decode("utf-8")
        except Exception as e:
            logger.error(f"Error getting file content {file_path}: {e}")
            raise

    def list_repository_files(self, path: str = "", branch: str = "main") -> List[str]:
        """List files in the repository."""
        try:
            contents = self.repo.get_contents(path, ref=branch)
            files = []
            for content in contents:
                if content.type == "file":
                    files.append(content.path)
                elif content.type == "dir":
                    files.extend(self.list_repository_files(content.path, branch))
            return files
        except Exception as e:
            logger.error(f"Error listing repository files: {e}")
            raise