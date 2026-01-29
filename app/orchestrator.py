"""Orchestrator for managing iterative SDLC cycles."""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .database import db_manager, IssueIteration, IterationStatus
from .github_client import get_github_client
from .github_app.auth import github_app_auth
from .config import settings
from .repository_analyzer import RepositoryAnalyzer
from .code_modifier import CodeModifier
from ai_coding_agent.code_agent import CodeAgent
from ai_coding_agent.reviewer_agent import ReviewerAgent
from ai_coding_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SDLCOrchestrator:
    """Orchestrator for managing the full SDLC cycle."""
    
    def __init__(self):
        # Initialize LLM client with settings
        self.llm_client = LLMClient(
            use_yandex_gpt=settings.use_yandex_gpt,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            yandex_gpt_api_key=settings.yandex_gpt_api_key,
            yandex_gpt_folder_id=settings.yandex_gpt_folder_id
        )
        
        # Note: GitHub client and agents will be created per-request with proper credentials
        # since they need installation-specific tokens
    
    async def start_issue_cycle(
        self,
        repo_full_name: str,
        issue_number: int,
        installation_id: int
    ) -> Optional[IssueIteration]:
        """Start a new SDLC cycle for an issue."""
        try:
            logger.info(f"Starting SDLC cycle for {repo_full_name}#{issue_number}")
            
            # Check if there's already an active iteration for this issue
            existing_iteration = db_manager.get_active_iteration(repo_full_name, issue_number)
            if existing_iteration:
                logger.info(f"Active iteration already exists for issue #{issue_number} (ID: {existing_iteration.id})")
                return existing_iteration
            
            # Get issue details
            owner, repo = repo_full_name.split("/")
            
            async with await get_github_client(installation_id) as github:
                issue_data = await github.get_issue(owner, repo, issue_number)
            
            # Create iteration record
            iteration = db_manager.create_iteration(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                installation_id=installation_id,
                issue_title=issue_data.get("title"),
                issue_body=issue_data.get("body"),
                max_iterations=settings.max_iterations
            )
            
            # Set consistent branch name for this issue
            branch_name = f"agent/issue-{issue_number}"
            db_manager.update_iteration(
                iteration.id,
                branch_name=branch_name
            )
            
            # Start first iteration
            await self._run_code_iteration(iteration)
            
            return iteration
            
        except Exception as e:
            logger.error(f"Failed to start issue cycle: {e}", exc_info=True)
            return None
    
    async def restart_issue_cycle(
        self,
        repo_full_name: str,
        issue_number: int,
        installation_id: int
    ) -> Optional[IssueIteration]:
        """Restart SDLC cycle for an issue (even if previous iterations failed)."""
        try:
            logger.info(f"Restarting SDLC cycle for {repo_full_name}#{issue_number}")
            
            # Mark any existing active iterations as failed
            existing_iteration = db_manager.get_active_iteration(repo_full_name, issue_number)
            if existing_iteration:
                logger.info(f"Marking existing iteration {existing_iteration.id} as failed to restart")
                db_manager.complete_iteration(existing_iteration.id, IterationStatus.FAILED)
            
            # Get issue details
            owner, repo = repo_full_name.split("/")
            
            async with await get_github_client(installation_id) as github:
                issue_data = await github.get_issue(owner, repo, issue_number)
            
            # Create new iteration record
            iteration = db_manager.create_iteration(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                installation_id=installation_id,
                issue_title=issue_data.get("title"),
                issue_body=issue_data.get("body"),
                max_iterations=settings.max_iterations
            )
            
            # Set consistent branch name for this issue
            branch_name = f"agent/issue-{issue_number}"
            db_manager.update_iteration(
                iteration.id,
                branch_name=branch_name
            )
            
            # Start first iteration
            await self._run_code_iteration(iteration)
            
            return iteration
            
        except Exception as e:
            logger.error(f"Failed to restart issue cycle: {e}", exc_info=True)
            return None
    
    async def _run_code_iteration(self, iteration: IssueIteration) -> bool:
        """Run a code generation iteration."""
        try:
            logger.info(f"Running code iteration {iteration.current_iteration + 1} for {iteration.repo_full_name}#{iteration.issue_number}")
            
            # Increment iteration counter
            iteration = db_manager.increment_iteration(iteration.id)
            if not iteration:
                logger.error("Failed to increment iteration")
                return False
            
            # Check if we've exceeded max iterations
            if iteration.current_iteration >= iteration.max_iterations:
                await self._complete_iteration(iteration, IterationStatus.FAILED, 
                                             "Maximum iterations reached")
                return False
            
            owner, repo = iteration.repo_full_name.split("/")
            
            # Prepare context for code agent
            context = {
                "repo_full_name": iteration.repo_full_name,
                "issue_number": iteration.issue_number,
                "issue_title": iteration.issue_title,
                "issue_body": iteration.issue_body,
                "iteration": iteration.current_iteration,
                "branch_name": iteration.branch_name,
                "pr_number": iteration.pr_number,
                "last_review_feedback": iteration.last_review_feedback
            }
            
            # Run code agent
            async with await get_github_client(iteration.installation_id) as github:
                result = await self._execute_code_agent(github, context, iteration)
            
            if not result:
                await self._complete_iteration(iteration, IterationStatus.FAILED,
                                             "Code generation failed")
                return False
            
            # Handle case where no PR was created (no changes)
            if result.get("no_changes"):
                await self._complete_iteration(iteration, IterationStatus.COMPLETED,
                                             "No changes needed - issue may already be resolved")
                return True
            
            # Update iteration with results
            db_manager.update_iteration(
                iteration.id,
                branch_name=result.get("branch_name"),
                pr_number=result.get("pr_number"),
                status=IterationStatus.WAITING_CI if result.get("pr_number") else IterationStatus.COMPLETED
            )
            
            if result.get("pr_number"):
                logger.info(f"Code iteration completed, waiting for CI. PR: {result.get('pr_number')}")
            else:
                logger.info(f"Code iteration completed with no PR created (no changes needed)")
                await self._complete_iteration(iteration, IterationStatus.COMPLETED,
                                             "Code changes applied but no PR needed")
            return True
            
        except Exception as e:
            logger.error(f"Code iteration failed: {e}", exc_info=True)
            await self._complete_iteration(iteration, IterationStatus.FAILED, str(e))
            return False
    
    async def _execute_code_agent(
        self,
        github,
        context: Dict[str, Any],
        iteration: IssueIteration
    ) -> Optional[Dict[str, Any]]:
        """Execute code agent with GitHub integration."""
        try:
            owner, repo = context["repo_full_name"].split("/")
            issue_number = context["issue_number"]
            
            # Always use consistent branch name for the same issue
            branch_name = f"agent/issue-{issue_number}"
            
            # If we have an existing branch_name in context, use it to maintain consistency
            if context.get("branch_name") and context["branch_name"] != branch_name:
                logger.warning(f"Branch name mismatch: context has {context['branch_name']}, expected {branch_name}")
                # Use the consistent naming pattern
            
            # Get default branch and its SHA
            default_branch = await github.get_default_branch(owner, repo)
            base_sha = await github.get_branch_sha(owner, repo, default_branch)
            
            # Create or update branch
            try:
                await github.create_branch(owner, repo, branch_name, base_sha)
                logger.info(f"Created branch {branch_name}")
            except Exception as e:
                # Check if it's a 422 error (branch already exists) or contains "already exists"
                error_str = str(e).lower()
                if "422" in error_str or "already exists" in error_str or "reference already exists" in error_str:
                    logger.info(f"Branch {branch_name} already exists, will update it")
                    # Try to update the existing branch to point to the latest commit
                    try:
                        await github.update_branch(owner, repo, branch_name, base_sha)
                        logger.info(f"Updated branch {branch_name} to latest commit")
                    except Exception as update_e:
                        logger.warning(f"Could not update branch {branch_name}: {update_e}, continuing anyway")
                else:
                    logger.error(f"Failed to create branch: {e}")
                    return None
            
            # Analyze issue and generate code changes with installation_id context
            context["installation_id"] = iteration.installation_id
            analysis = await self._analyze_issue_requirements(context)
            if not analysis:
                return None
            
            # Apply code changes
            changes_applied = await self._apply_code_changes(
                github, owner, repo, branch_name, analysis, context
            )
            
            if not changes_applied:
                return None
            
            # Create or update PR - always try to reuse existing PR for the same issue
            pr_number = context.get("pr_number")
            
            # First, check if there's already a PR for this branch
            if not pr_number:
                try:
                    existing_prs = await github.list_pull_requests(owner, repo, head=f"{owner}:{branch_name}", state="open")
                    if existing_prs:
                        pr_number = existing_prs[0]["number"]
                        logger.info(f"Found existing open PR #{pr_number} for branch {branch_name}")
                except Exception as e:
                    logger.warning(f"Failed to check for existing PRs: {e}")
            
            if pr_number:
                # Update existing PR
                pr_title = f"Fix #{issue_number}: {context['issue_title']} (Iteration {context['iteration']})"
                pr_body = self._generate_pr_description(context, analysis)
                
                try:
                    await github.update_pull_request(
                        owner, repo, pr_number, title=pr_title, body=pr_body
                    )
                    logger.info(f"Updated existing PR #{pr_number}")
                except Exception as e:
                    logger.error(f"Failed to update PR #{pr_number}: {e}")
                    # Continue anyway, the PR might still be valid
            else:
                # Check if branch has any changes compared to base before creating new PR
                try:
                    branch_sha = await github.get_branch_sha(owner, repo, branch_name)
                    base_sha = await github.get_branch_sha(owner, repo, default_branch)
                    
                    if branch_sha == base_sha:
                        logger.warning(f"Branch {branch_name} has no changes compared to {default_branch}, skipping PR creation")
                        # Return success but without PR number to indicate no PR was needed
                        return {
                            "branch_name": branch_name,
                            "pr_number": None,
                            "analysis": analysis,
                            "no_changes": True
                        }
                    
                    # Create new PR
                    pr_title = f"Fix #{issue_number}: {context['issue_title']}"
                    pr_body = self._generate_pr_description(context, analysis)
                    
                    pr_data = await github.create_pull_request(
                        owner, repo, pr_title, pr_body, branch_name, default_branch
                    )
                    pr_number = pr_data["number"]
                    logger.info(f"Created new PR #{pr_number}")
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "422" in error_str and ("no commits" in error_str or "no changes" in error_str or "already exists" in error_str):
                        logger.warning(f"Cannot create PR: {e}")
                        # Try one more time to find existing PR
                        try:
                            existing_prs = await github.list_pull_requests(owner, repo, head=f"{owner}:{branch_name}")
                            if existing_prs:
                                pr_number = existing_prs[0]["number"]
                                logger.info(f"Found existing PR #{pr_number} for branch {branch_name} after creation failure")
                            else:
                                logger.error(f"No changes to create PR and no existing PR found")
                                return None
                        except Exception as list_e:
                            logger.error(f"Failed to check for existing PRs: {list_e}")
                            return None
                    else:
                        logger.error(f"Failed to create PR: {e}")
                        return None
            
            return {
                "branch_name": branch_name,
                "pr_number": pr_number,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Code agent execution failed: {e}", exc_info=True)
            return None
    
    async def _analyze_issue_requirements(self, context: Dict[str, Any]) -> Optional[Dict]:
        """Analyze issue requirements using LLM with repository architecture awareness."""
        try:
            # Get repository structure analysis
            owner, repo = context["repo_full_name"].split("/")
            
            # Initialize repository analyzer
            repository_analyzer = RepositoryAnalyzer()
            
            # Get GitHub client for repository analysis
            async with await get_github_client(context.get("installation_id")) as github:
                # Analyze repository structure
                logger.info(f"Analyzing repository structure for {owner}/{repo}")
                repository_map = await repository_analyzer.analyze_repository(github, owner, repo)
                
                if not repository_map:
                    logger.warning("Could not analyze repository structure, falling back to basic analysis")
                    return await self._analyze_issue_requirements_basic(context)
                
                # Find relevant existing classes and functions
                issue_description = f"{context['issue_title']} {context['issue_body'] or ''}"
                relevant_classes = repository_analyzer.find_relevant_classes(repository_map, issue_description)
                modification_targets = repository_analyzer.suggest_modification_targets(repository_map, issue_description)
                
                logger.info(f"Found {len(relevant_classes)} relevant classes and {len(modification_targets)} modification targets")
                
                # Build architecture-aware system prompt
                system_prompt = f"""You are an expert software developer analyzing GitHub issues with FULL REPOSITORY CONTEXT.

EXISTING REPOSITORY STRUCTURE:
{repository_map.get_structure_summary()}

RELEVANT EXISTING CLASSES:
{self._format_relevant_classes(relevant_classes)}

SUGGESTED MODIFICATION TARGETS:
{self._format_modification_targets(modification_targets)}

ARCHITECTURE PATTERNS DETECTED:
{', '.join(repository_map.architecture_patterns) if repository_map.architecture_patterns else 'Standard Python project'}

CRITICAL RULES:
1. **ALWAYS prefer modifying existing classes over creating new ones**
2. **If a class already exists for the functionality, specify it in files_to_modify, NOT files_to_create**
3. **Only create new files if absolutely necessary and no existing class can be enhanced**
4. **Follow the existing architecture patterns and naming conventions**
5. **Understand the existing code structure and build upon it**
6. **When modifying existing classes, preserve existing functionality**

MODIFICATION STRATEGY:
- If you need to add functionality to an existing class, put the file in files_to_modify
- If you need to create a completely new component that doesn't exist, put it in files_to_create
- Always check if similar functionality already exists before creating new files

Analyze the issue and provide a structured response for implementation.

Respond in JSON format:
{{
    "summary": "Brief description of what will be implemented",
    "files_to_modify": ["list", "of", "existing", "files", "to", "modify"],
    "files_to_create": ["list", "of", "new", "files", "only", "if", "necessary"],
    "requirements": ["list", "of", "requirements"],
    "technical_approach": "Implementation approach that builds on existing architecture",
    "dependencies": ["list", "of", "dependencies"],
    "architecture_notes": "How this fits into the existing architecture",
    "existing_classes_to_enhance": ["list", "of", "existing", "classes", "that", "will", "be", "enhanced"]
}}"""
                
                user_prompt = f"""Issue #{context['issue_number']}: {context['issue_title']}

Description:
{context['issue_body'] or 'No description provided'}

Iteration: {context['iteration']}

REPOSITORY CONTEXT:
- Total Python files: {len(repository_map.modules)}
- Main classes: {', '.join([cls.name for cls in relevant_classes[:5]])}
- Architecture: {', '.join(repository_map.architecture_patterns) if repository_map.architecture_patterns else 'Standard Python'}"""
                
                if context.get('last_review_feedback'):
                    user_prompt += f"\n\nPrevious review feedback:\n{context['last_review_feedback']}"
                
                messages = [
                    self.llm_client.create_system_message(system_prompt),
                    self.llm_client.create_user_message(user_prompt)
                ]
                
                response = await self.llm_client.generate_response(messages)
                
                # Parse JSON response
                import json
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Add repository context to analysis
                    analysis["repository_map"] = repository_map
                    analysis["relevant_classes"] = relevant_classes
                    analysis["modification_targets"] = modification_targets
                    
                    # Log the analysis for debugging
                    logger.info(f"Architecture-aware analysis completed:")
                    logger.info(f"  - Files to modify: {analysis.get('files_to_modify', [])}")
                    logger.info(f"  - Files to create: {analysis.get('files_to_create', [])}")
                    logger.info(f"  - Existing classes to enhance: {analysis.get('existing_classes_to_enhance', [])}")
                    
                    return analysis
                
                logger.error("Could not parse LLM analysis response")
                return None
            
        except Exception as e:
            logger.error(f"Architecture-aware issue analysis failed: {e}", exc_info=True)
            # Fallback to basic analysis
            logger.info("Falling back to basic issue analysis")
            return await self._analyze_issue_requirements_basic(context)
    
    async def _analyze_issue_requirements_basic(self, context: Dict[str, Any]) -> Optional[Dict]:
        """Fallback basic issue analysis without repository context."""
        try:
            system_prompt = """You are an expert software developer analyzing GitHub issues.
            Analyze the issue and provide a structured response for implementation.
            
            Consider:
            1. What needs to be implemented
            2. Files to create or modify
            3. Technical approach
            4. Dependencies needed
            
            If this is a follow-up iteration, also consider the previous feedback.
            
            Respond in JSON format:
            {
                "summary": "Brief description",
                "files_to_modify": ["list", "of", "files"],
                "files_to_create": ["list", "of", "new", "files"],
                "requirements": ["list", "of", "requirements"],
                "technical_approach": "Implementation approach",
                "dependencies": ["list", "of", "dependencies"]
            }"""
            
            user_prompt = f"""Issue #{context['issue_number']}: {context['issue_title']}

Description:
{context['issue_body'] or 'No description provided'}

Iteration: {context['iteration']}"""
            
            if context.get('last_review_feedback'):
                user_prompt += f"\n\nPrevious review feedback:\n{context['last_review_feedback']}"
            
            messages = [
                self.llm_client.create_system_message(system_prompt),
                self.llm_client.create_user_message(user_prompt)
            ]
            
            response = await self.llm_client.generate_response(messages)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            logger.error("Could not parse LLM analysis response")
            return None
            
        except Exception as e:
            logger.error(f"Basic issue analysis failed: {e}", exc_info=True)
            return None
    
    def _format_relevant_classes(self, relevant_classes) -> str:
        """Format relevant classes for LLM prompt."""
        if not relevant_classes:
            return "No directly relevant classes found."
        
        formatted = []
        for cls in relevant_classes[:10]:  # Limit to top 10 to avoid token limits
            methods_str = ", ".join(cls.methods[:5])  # Show first 5 methods
            if len(cls.methods) > 5:
                methods_str += f" (and {len(cls.methods) - 5} more)"
            
            formatted.append(f"- {cls.name} in {cls.file_path}")
            formatted.append(f"  Purpose: {cls.purpose}")
            formatted.append(f"  Methods: {methods_str}")
            formatted.append(f"  Dependencies: {', '.join(cls.dependencies[:3])}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_modification_targets(self, modification_targets) -> str:
        """Format modification targets for LLM prompt."""
        if not modification_targets:
            return "No specific modification targets identified."
        
        formatted = []
        for target in modification_targets[:5]:  # Limit to top 5
            formatted.append(f"- {target.target_type}: {target.name} in {target.file_path}")
            formatted.append(f"  Reason: {target.reason}")
            formatted.append(f"  Confidence: {target.confidence}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    async def _apply_code_changes(
        self,
        github,
        owner: str,
        repo: str,
        branch_name: str,
        analysis: Dict,
        context: Dict[str, Any]
    ) -> bool:
        """Apply code changes to the repository."""
        try:
            # Get repository structure
            repo_files = await github.list_repository_files(owner, repo)
            
            # Process files to modify
            for file_path in analysis.get("files_to_modify", []):
                success = await self._modify_file(
                    github, owner, repo, branch_name, file_path, analysis, context
                )
                if not success:
                    logger.warning(f"Failed to modify {file_path}")
            
            # Process files to create
            for file_path in analysis.get("files_to_create", []):
                success = await self._create_file(
                    github, owner, repo, branch_name, file_path, analysis, context
                )
                if not success:
                    logger.warning(f"Failed to create {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply code changes: {e}", exc_info=True)
            return False
    
    async def _modify_file(
        self,
        github,
        owner: str,
        repo: str,
        branch_name: str,
        file_path: str,
        analysis: Dict,
        context: Dict[str, Any]
    ) -> bool:
        """Modify an existing file."""
        try:
            # Get current file content
            file_data = await github.get_file_content(owner, repo, file_path, branch_name)
            if not file_data:
                logger.warning(f"File {file_path} not found, will create instead")
                return await self._create_file(github, owner, repo, branch_name, file_path, analysis, context)
            
            import base64
            current_content = base64.b64decode(file_data["content"]).decode()
            
            # Generate modified content using LLM
            modified_content = await self._generate_file_content(
                file_path, current_content, analysis, context, is_modification=True
            )
            
            if not modified_content:
                return False
            
            # Check if content actually changed
            if current_content.strip() == modified_content.strip():
                logger.info(f"File {file_path} content unchanged, skipping modification")
                return True
            
            # Update file
            commit_message = f"Modify {file_path} for issue #{context['issue_number']} (iteration {context['iteration']})"
            
            try:
                await github.create_or_update_file(
                    owner, repo, file_path, modified_content, commit_message, branch_name, file_data["sha"]
                )
                logger.info(f"Modified file {file_path}")
                return True
            except Exception as update_e:
                error_str = str(update_e).lower()
                if "422" in error_str and ("sha" in error_str or "does not match" in error_str):
                    logger.warning(f"SHA mismatch for {file_path}, refetching and retrying")
                    # Refetch file data to get latest SHA
                    fresh_file_data = await github.get_file_content(owner, repo, file_path, branch_name)
                    if fresh_file_data:
                        await github.create_or_update_file(
                            owner, repo, file_path, modified_content, commit_message, branch_name, fresh_file_data["sha"]
                        )
                        logger.info(f"Modified file {file_path} after SHA refresh")
                        return True
                raise update_e
            
        except Exception as e:
            logger.error(f"Failed to modify file {file_path}: {e}", exc_info=True)
            return False
    
    async def _create_file(
        self,
        github,
        owner: str,
        repo: str,
        branch_name: str,
        file_path: str,
        analysis: Dict,
        context: Dict[str, Any]
    ) -> bool:
        """Create a new file."""
        try:
            # Check if file already exists on the branch
            existing_file = await github.get_file_content(owner, repo, file_path, branch_name)
            
            # Generate file content using LLM
            file_content = await self._generate_file_content(
                file_path, None, analysis, context, is_modification=False
            )
            
            if not file_content:
                return False
            
            # If file exists, check if content is different
            if existing_file:
                import base64
                existing_content = base64.b64decode(existing_file["content"]).decode()
                if existing_content.strip() == file_content.strip():
                    logger.info(f"File {file_path} already exists with same content, skipping")
                    return True
                
                # File exists but content is different, update it instead
                logger.info(f"File {file_path} exists but content differs, updating instead of creating")
                return await self._modify_file(github, owner, repo, branch_name, file_path, analysis, context)
            
            # Create file
            commit_message = f"Create {file_path} for issue #{context['issue_number']} (iteration {context['iteration']})"
            
            await github.create_or_update_file(
                owner, repo, file_path, file_content, commit_message, branch_name
            )
            
            logger.info(f"Created file {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}", exc_info=True)
            return False
    
    async def _generate_file_content(
        self,
        file_path: str,
        current_content: Optional[str],
        analysis: Dict,
        context: Dict[str, Any],
        is_modification: bool
    ) -> Optional[str]:
        """Generate file content using LLM with architecture awareness."""
        try:
            # Check if we have repository context from analysis
            repository_map = analysis.get("repository_map")
            relevant_classes = analysis.get("relevant_classes", [])
            modification_targets = analysis.get("modification_targets", [])
            
            # Use CodeModifier for intelligent modifications if we have repository context
            if is_modification and repository_map and current_content:
                logger.info(f"Using architecture-aware modification for {file_path}")
                
                # Initialize CodeModifier
                code_modifier = CodeModifier()
                
                # Find if this file has relevant classes to modify
                file_classes = [cls for cls in relevant_classes if cls.file_path == file_path]
                file_targets = [target for target in modification_targets if target.file_path == file_path]
                
                if file_classes or file_targets:
                    # Use intelligent code modification
                    modified_content = await code_modifier.modify_existing_file(
                        file_path=file_path,
                        current_content=current_content,
                        analysis=analysis,
                        repository_map=repository_map,
                        llm_client=self.llm_client
                    )
                    
                    if modified_content:
                        logger.info(f"Successfully used architecture-aware modification for {file_path}")
                        return modified_content
                    else:
                        logger.warning(f"Architecture-aware modification failed for {file_path}, falling back to basic modification")
            
            # Build architecture context for LLM prompt
            architecture_context = ""
            if repository_map:
                architecture_context = f"""
REPOSITORY ARCHITECTURE CONTEXT:
- Architecture patterns: {', '.join(repository_map.architecture_patterns) if repository_map.architecture_patterns else 'Standard Python'}
- Total modules: {len(repository_map.modules)}
- Existing classes in this file: {', '.join([cls.name for cls in relevant_classes if cls.file_path == file_path])}

ARCHITECTURE GUIDELINES:
- Follow existing naming conventions and patterns
- Maintain consistency with existing code structure
- Preserve existing functionality when modifying
- Use similar import patterns as other files in the project"""
            
            if is_modification:
                system_prompt = f"""You are an expert software developer modifying code files with full repository context.
                
                Modify the existing file to implement the required functionality while preserving existing architecture.
                
                Requirements:
                - Summary: {analysis.get('summary', '')}
                - Technical approach: {analysis.get('technical_approach', '')}
                - Requirements: {', '.join(analysis.get('requirements', []))}
                - Architecture notes: {analysis.get('architecture_notes', '')}
                
                {architecture_context}
                
                CRITICAL RULES:
                1. **PRESERVE all existing functionality unless explicitly conflicting**
                2. **Follow the existing code style and patterns in this file**
                3. **Add new functionality by enhancing existing classes/functions when possible**
                4. **Maintain existing imports and add new ones as needed**
                5. **Keep existing method signatures unless modification is required**
                6. **Add proper error handling and documentation**
                7. **Include type hints where appropriate**
                
                Return only the complete modified file content."""
                
                user_prompt = f"""File to modify: {file_path}

Current content:
```
{current_content}
```

Existing classes to enhance: {', '.join(analysis.get('existing_classes_to_enhance', []))}

Please provide the modified file content that enhances the existing code."""
            else:
                system_prompt = f"""You are an expert software developer creating new code files with repository context.
                
                Create a new file that implements the required functionality and fits into the existing architecture.
                
                Requirements:
                - Summary: {analysis.get('summary', '')}
                - Technical approach: {analysis.get('technical_approach', '')}
                - Requirements: {', '.join(analysis.get('requirements', []))}
                - Dependencies: {', '.join(analysis.get('dependencies', []))}
                - Architecture notes: {analysis.get('architecture_notes', '')}
                
                {architecture_context}
                
                RULES:
                1. **Follow the existing architecture patterns and naming conventions**
                2. **Use similar import patterns as other files in the project**
                3. **Follow best practices and coding standards**
                4. **Add comprehensive documentation**
                5. **Include proper error handling**
                6. **Add type hints and imports**
                7. **Ensure the new file integrates well with existing code**
                
                Return only the complete file content."""
                
                user_prompt = f"""Create new file: {file_path}

This file should integrate with existing architecture patterns.

Please provide the complete file content."""
            
            if context.get('last_review_feedback'):
                user_prompt += f"\n\nConsider this feedback from previous review:\n{context['last_review_feedback']}"
            
            messages = [
                self.llm_client.create_system_message(system_prompt),
                self.llm_client.create_user_message(user_prompt)
            ]
            
            response = await self.llm_client.generate_response(messages)
            
            # Clean up response (remove code block markers)
            import re
            response = re.sub(r'^```[a-zA-Z]*\n', '', response)
            response = re.sub(r'\n```$', '', response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate content for {file_path}: {e}", exc_info=True)
            return None
    
    def _generate_pr_description(self, context: Dict[str, Any], analysis: Dict) -> str:
        """Generate PR description."""
        description = f"""## Fixes #{context['issue_number']}

**Issue Title:** {context['issue_title']}

**Iteration:** {context['iteration']}/{context.get('max_iterations', settings.max_iterations)}

**Summary:** {analysis.get('summary', 'No summary available')}

### Changes Made:
"""
        
        if analysis.get('files_to_create'):
            description += "\n**New Files:**\n"
            for file_path in analysis['files_to_create']:
                description += f"- `{file_path}`\n"
        
        if analysis.get('files_to_modify'):
            description += "\n**Modified Files:**\n"
            for file_path in analysis['files_to_modify']:
                description += f"- `{file_path}`\n"

        if analysis.get('requirements'):
            description += "\n**Requirements Implemented:**\n"
            for req in analysis['requirements']:
                description += f"- {req}\n"

        if analysis.get('technical_approach'):
            description += f"\n**Technical Approach:**\n{analysis['technical_approach']}\n"

        if context.get('last_review_feedback'):
            description += f"\n**Addressed Feedback:**\n{context['last_review_feedback']}\n"

        description += "\n### Testing\n"
        description += "- [ ] Code follows project standards\n"
        description += "- [ ] All tests pass\n"
        description += "- [ ] No linting errors\n"
        description += "- [ ] Functionality works as expected\n"

        return description
    
    async def handle_ci_completion(
        self,
        repo_full_name: str,
        pr_number: int,
        ci_status: str,
        ci_conclusion: str,
        head_sha: str = None
    ) -> bool:
        """Handle CI completion event with SHA binding."""
        try:
            sha_info = f"@{head_sha[:8]}" if head_sha else ""
            logger.info(f"Handling CI completion for {repo_full_name} PR#{pr_number}{sha_info}: {ci_status}/{ci_conclusion}")
            
            # Find iteration by PR
            iteration = db_manager.get_iteration_by_pr(repo_full_name, pr_number)
            if not iteration:
                logger.warning(f"No active iteration found for PR #{pr_number}")
                return False
            
            # If we have a head SHA, validate it matches current PR head
            if head_sha:
                # Get current PR head SHA to validate
                try:
                    owner, repo = repo_full_name.split("/")
                    async with await get_github_client(iteration.installation_id) as github:
                        pr_data = await github.get_pull_request(owner, repo, pr_number)
                        current_head_sha = pr_data.get("head", {}).get("sha")
                        
                        if current_head_sha and current_head_sha != head_sha:
                            logger.warning(f"CI completion SHA {head_sha[:8]} doesn't match current PR head {current_head_sha[:8]}, ignoring stale CI event")
                            return False
                        
                        logger.info(f"CI completion SHA {head_sha[:8]} matches current PR head, proceeding")
                except Exception as e:
                    logger.warning(f"Could not validate PR head SHA: {e}, proceeding anyway")
            
            # Check if already reviewing to prevent duplicate reviews
            if iteration.status == IterationStatus.REVIEWING:
                logger.info(f"Review already in progress for iteration {iteration.id}, skipping")
                return True
            
            # Only proceed if waiting for CI
            if iteration.status != IterationStatus.WAITING_CI:
                logger.info(f"Iteration {iteration.id} not waiting for CI (status: {iteration.status}), skipping")
                return True
            
            # Update CI status with SHA binding
            logger.info(f"Updating iteration {iteration.id} with CI status: {ci_status}, conclusion: {ci_conclusion}, SHA: {head_sha[:8] if head_sha else 'unknown'}")
            
            if head_sha:
                # Use new SHA-aware update method
                db_manager.update_ci_status(
                    iteration.id,
                    ci_status=ci_status,
                    ci_conclusion=ci_conclusion,
                    pr_head_sha=head_sha,
                    source="webhook"
                )
            else:
                # Fallback to old method for backward compatibility
                db_manager.update_iteration(
                    iteration.id,
                    last_ci_status=ci_status,
                    last_ci_conclusion=ci_conclusion,
                    status=IterationStatus.REVIEWING
                )
            
            # Update status to reviewing
            db_manager.update_iteration(iteration.id, status=IterationStatus.REVIEWING)
            
            # Verify the update
            updated_iteration = db_manager.get_iteration_by_pr(repo_full_name, pr_number)
            if updated_iteration:
                logger.info(f"Verified iteration {updated_iteration.id} CI status: {updated_iteration.last_ci_status}, conclusion: {updated_iteration.last_ci_conclusion}, SHA: {updated_iteration.pr_head_sha[:8] if updated_iteration.pr_head_sha else 'none'}")
            else:
                logger.error(f"Could not find updated iteration for PR #{pr_number}")
            
            # Start review process
            await self._run_review_iteration(updated_iteration or iteration)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle CI completion: {e}", exc_info=True)
            return False
    
    async def _run_review_iteration(self, iteration: IssueIteration) -> bool:
        """Run review iteration with SHA-aware CI status checking."""
        try:
            logger.info(f"Running review for {iteration.repo_full_name}#{iteration.issue_number}")
            logger.info(f"Iteration {iteration.id} current CI status: {iteration.last_ci_status}, conclusion: {iteration.last_ci_conclusion}, SHA: {iteration.pr_head_sha[:8] if iteration.pr_head_sha else 'none'}")
            
            owner, repo = iteration.repo_full_name.split("/")
            
            async with await get_github_client(iteration.installation_id) as github:
                # Get PR details and files
                pr_data = await github.get_pull_request(owner, repo, iteration.pr_number)
                pr_files = await github.get_pull_request_files(owner, repo, iteration.pr_number)
                current_head_sha = pr_data.get("head", {}).get("sha")
                
                # Get detailed CI status with SHA validation
                ci_status_details = None
                if current_head_sha:
                    logger.info(f"Getting CI status for current PR head SHA {current_head_sha[:8]}")
                    ci_status_details = await github.get_pr_ci_status(
                        owner, repo, iteration.pr_number, expected_sha=current_head_sha
                    )
                else:
                    logger.warning("Could not get current PR head SHA")
                    ci_status_details = await github.get_pr_ci_status(owner, repo, iteration.pr_number)
                
                # Determine actual CI conclusion using DB-first logic
                actual_ci_conclusion = "unknown"
                ci_data_source = "unknown"
                
                # Check if we have recent webhook data for the current SHA
                if (iteration.pr_head_sha and current_head_sha and
                    iteration.pr_head_sha == current_head_sha and
                    iteration.ci_status_source == "webhook"):
                    
                    from datetime import datetime, timedelta
                    if (iteration.ci_status_updated_at and
                        datetime.utcnow() - iteration.ci_status_updated_at < timedelta(minutes=5)):
                        actual_ci_conclusion = iteration.last_ci_conclusion
                        ci_data_source = "webhook (recent)"
                        logger.info(f"Using recent webhook CI data for SHA {current_head_sha[:8]}: {actual_ci_conclusion}")
                    else:
                        logger.info(f"Webhook CI data is stale, using API data")
                
                # Fallback to API data
                if actual_ci_conclusion == "unknown" and ci_status_details:
                    if ci_status_details.get("overall_conclusion") and ci_status_details["overall_conclusion"] != "unknown":
                        actual_ci_conclusion = ci_status_details["overall_conclusion"]
                        ci_data_source = "api"
                        logger.info(f"Using API CI data for SHA {current_head_sha[:8] if current_head_sha else 'unknown'}: {actual_ci_conclusion}")
                
                # Final fallback to stored data
                if actual_ci_conclusion == "unknown":
                    actual_ci_conclusion = iteration.last_ci_conclusion or "unknown"
                    ci_data_source = "stored"
                    logger.info(f"Using stored CI data: {actual_ci_conclusion}")
                
                logger.info(f"Final CI status for review: {actual_ci_conclusion} (source: {ci_data_source})")
                
                # Update ci_status_details to reflect the chosen conclusion
                if ci_status_details:
                    ci_status_details["overall_conclusion"] = actual_ci_conclusion
                    ci_status_details["overall_status"] = "completed" if actual_ci_conclusion in ["success", "failure"] else "pending"
                    ci_status_details["data_source"] = ci_data_source
                
                # Prepare review context
                review_context = {
                    "repo_full_name": iteration.repo_full_name,
                    "issue_number": iteration.issue_number,
                    "issue_title": iteration.issue_title,
                    "issue_body": iteration.issue_body,
                    "pr_number": iteration.pr_number,
                    "pr_data": pr_data,
                    "pr_files": pr_files,
                    "ci_status": iteration.last_ci_status,
                    "ci_conclusion": actual_ci_conclusion,
                    "ci_status_details": ci_status_details,
                    "ci_data_source": ci_data_source,
                    "iteration": iteration.current_iteration,
                    "installation_id": iteration.installation_id
                }
                
                # Run review
                review_result = await self._execute_reviewer_agent(review_context)
                
                if not review_result:
                    await self._complete_iteration(iteration, IterationStatus.FAILED, "Review failed")
                    return False
                
                # Update iteration with review results
                overall_assessment = review_result.get("overall_assessment", {})
                db_manager.update_iteration(
                    iteration.id,
                    last_review_score=overall_assessment.get("score"),
                    last_review_recommendation=overall_assessment.get("recommendation"),
                    last_review_feedback=overall_assessment.get("summary", "")
                )
                
                # Post review to PR
                await self._post_review_results(github, review_context, review_result)
                
                # Decide next action
                await self._decide_next_action(iteration, review_result, review_context)
                
                return True
                
        except Exception as e:
            logger.error(f"Review iteration failed: {e}", exc_info=True)
            await self._complete_iteration(iteration, IterationStatus.FAILED, str(e))
            return False
    
    async def _execute_reviewer_agent(self, context: Dict[str, Any]) -> Optional[Dict]:
        """Execute reviewer agent with architecture awareness."""
        try:
            # Create GitHub client for this installation
            async with await get_github_client(context.get("installation_id")) as github:
                # Get repository architecture context for review
                owner, repo = context["repo_full_name"].split("/")
                
                # Initialize repository analyzer for architecture context
                repository_analyzer = RepositoryAnalyzer()
                repository_map = None
                
                try:
                    logger.info(f"Analyzing repository structure for review context: {owner}/{repo}")
                    repository_map = await repository_analyzer.analyze_repository(github, owner, repo)
                    if repository_map:
                        logger.info(f"Repository analysis successful for review: {len(repository_map.modules)} modules found")
                    else:
                        logger.warning("Repository analysis failed for review, proceeding without architecture context")
                except Exception as e:
                    logger.warning(f"Repository analysis failed for review: {e}, proceeding without architecture context")
                
                # Create GitHub client wrapper for old agent
                from ai_coding_agent.github_client import GitHubClient
                
                # Get installation token for this repo
                installation_token = await github_app_auth.get_installation_token(
                    context.get("installation_id")
                )
                
                # Create old-style GitHub client
                github_client = GitHubClient(
                    github_token=installation_token,
                    repo_owner=owner,
                    repo_name=repo
                )
                
                # Create reviewer agent with dependencies
                reviewer_agent = ReviewerAgent(github_client, self.llm_client)
                
                # Prepare review data similar to existing reviewer agent
                pr_files_data = []
                for file_info in context["pr_files"]:
                    pr_files_data.append({
                        "filename": file_info["filename"],
                        "status": file_info["status"],
                        "additions": file_info["additions"],
                        "deletions": file_info["deletions"],
                        "changes": file_info["changes"],
                        "patch": file_info.get("patch", "")
                    })
                
                # Use existing reviewer agent logic
                # Create proper issue data structure that the reviewer agent expects
                # The reviewer agent expects an object with .title and .body attributes
                class IssueData:
                    def __init__(self, title, body, number):
                        self.title = title
                        self.body = body
                        self.number = number
                
                issue_data = IssueData(
                    title=context["issue_title"],
                    body=context["issue_body"] or "",
                    number=context["issue_number"]
                )
                
                # Perform comprehensive review with architecture context
                review_result = await reviewer_agent._perform_comprehensive_review(
                    context["pr_data"],
                    issue_data,
                    pr_files_data
                )
                
                # Add architecture awareness to review if we have repository context
                if repository_map:
                    logger.info("Adding architecture awareness to review results")
                    
                    # Analyze changed files for architecture compliance
                    architecture_issues = []
                    modified_files = [f["filename"] for f in pr_files_data if f["status"] in ["modified", "added"]]
                    
                    for file_path in modified_files:
                        if file_path.endswith('.py'):
                            # Check if file follows existing patterns
                            similar_files = [m for m in repository_map.modules.values()
                                           if m.file_path != file_path and
                                           m.file_path.split('/')[-1].split('.')[0] in file_path]
                            
                            if similar_files:
                                # File should follow similar patterns
                                pattern_file = similar_files[0]
                                architecture_issues.append(f"File {file_path} should follow patterns from {pattern_file.file_path}")
                    
                    # Add architecture assessment to review
                    if not review_result.get("architecture_analysis"):
                        review_result["architecture_analysis"] = {}
                    
                    review_result["architecture_analysis"]["repository_context"] = {
                        "total_modules": len(repository_map.modules),
                        "architecture_patterns": repository_map.architecture_patterns,
                        "modified_files_count": len(modified_files),
                        "architecture_issues": architecture_issues
                    }
                    
                    # Adjust score based on architecture compliance
                    if architecture_issues:
                        penalty = min(10, len(architecture_issues) * 3)  # Max 10 point penalty
                        if review_result.get("overall_assessment", {}).get("score"):
                            review_result["overall_assessment"]["score"] = max(0,
                                review_result["overall_assessment"]["score"] - penalty)
                        
                        # Add architecture issues to summary
                        arch_summary = f" Architecture issues found: {len(architecture_issues)} patterns not followed."
                        if review_result.get("overall_assessment", {}).get("summary"):
                            review_result["overall_assessment"]["summary"] += arch_summary
                        else:
                            review_result["overall_assessment"]["summary"] = arch_summary.strip()
                
                # Add CI status consideration
                ci_conclusion = context.get("ci_conclusion")
                if not ci_conclusion or ci_conclusion == "unknown":
                    # Try to get from detailed CI status
                    ci_details = context.get("ci_status_details", {})
                    if ci_details.get("overall_conclusion"):
                        ci_conclusion = ci_details["overall_conclusion"]
                
                if ci_conclusion and ci_conclusion != "success" and ci_conclusion != "unknown":
                    # Penalize score for failing CI
                    if review_result.get("overall_assessment", {}).get("score", 0) > 50:
                        review_result["overall_assessment"]["score"] *= 0.8
                        review_result["overall_assessment"]["summary"] += f" CI failed with status: {ci_conclusion}"
                
                return review_result
            
        except Exception as e:
            logger.error(f"Reviewer agent execution failed: {e}", exc_info=True)
            return None
    
    async def _post_review_results(
        self, 
        github, 
        context: Dict[str, Any], 
        review_result: Dict
    ) -> None:
        """Post review results to PR."""
        try:
            owner, repo = context["repo_full_name"].split("/")
            
            # Generate review comment
            comment = self._format_review_comment(review_result, context)
            
            # Post comment
            await github.create_issue_comment(
                owner, repo, context["pr_number"], comment
            )
            
            # Determine review event
            recommendation = review_result.get("overall_assessment", {}).get("recommendation", "")
            if recommendation == "approve" and context["ci_conclusion"] == "success":
                event = "APPROVE"
            elif recommendation == "request_changes":
                event = "REQUEST_CHANGES"
            else:
                event = "COMMENT"
            
            # Create review
            review_summary = review_result.get("overall_assessment", {}).get("summary", "")
            
            # Ensure review summary is not empty
            if not review_summary or review_summary.strip() == "":
                review_summary = "Automated code review completed."
            
            try:
                await github.create_pull_request_review(
                    owner, repo, context["pr_number"], review_summary, event
                )
            except Exception as review_e:
                error_str = str(review_e).lower()
                if "422" in error_str:
                    logger.warning(f"Failed to create PR review, possibly due to empty content or duplicate review: {review_e}")
                    # Continue without failing the entire process
                else:
                    raise review_e
            
            logger.info(f"Posted review results to PR #{context['pr_number']}")
            
        except Exception as e:
            logger.error(f"Failed to post review results: {e}", exc_info=True)
    
    def _format_review_comment(self, review_result: Dict, context: Dict[str, Any]) -> str:
        """Format review comment with enhanced SHA-specific logging."""
        overall = review_result.get("overall_assessment", {})
        
        # Get SHA and data source info for enhanced logging
        ci_details = context.get('ci_status_details', {})
        head_sha = ci_details.get('head_sha', 'unknown')
        data_source = context.get('ci_data_source', 'unknown')
        
        comment = f"""## 🤖 AI Code Review - Iteration {context['iteration']}

### {overall.get('status', 'Review Completed')}

**Overall Score:** {overall.get('score', 0)}/100

**CI Status:** {context.get('ci_conclusion', 'unknown')} ({'✅' if context.get('ci_conclusion') == 'success' else '❌'})
**CI Data Source:** {data_source}
**Commit SHA:** `{head_sha[:8] if head_sha != 'unknown' else 'unknown'}`

**Detailed CI Status:**"""

        # Add detailed CI information if available
        if ci_details and ci_details.get('overall_conclusion') and ci_details.get('overall_conclusion') != 'unknown':
            actual_ci_status = ci_details['overall_conclusion']
            comment = comment.replace(
                f"**CI Status:** {context.get('ci_conclusion', 'unknown')}",
                f"**CI Status:** {actual_ci_status}"
            )
            comment = comment.replace(
                f"({'✅' if context.get('ci_conclusion') == 'success' else '❌'})",
                f"({'✅' if actual_ci_status == 'success' else '❌'})"
            )
            
            comment += f"""
- Overall: {ci_details['overall_conclusion']} ({ci_details.get('overall_status', 'unknown')})"""
            
            # Show required vs total checks info
            required_checks = ci_details.get('required_checks', [])
            total_checks = len(ci_details.get('check_runs', []))
            if required_checks:
                comment += f"""
- Required Checks: {len(required_checks)} of {total_checks} total checks"""
            
            if ci_details.get('failed_checks'):
                comment += f"""
- Failed Checks: {', '.join(ci_details['failed_checks'])}"""
            
            if ci_details.get('pending_checks'):
                comment += f"""
- Pending Checks: {', '.join(ci_details['pending_checks'])}"""
        else:
            comment += """
- Status information not available"""

        comment += f"""

### 📊 Analysis:
- **Code Quality:** {review_result.get('code_quality', {}).get('summary', 'N/A')}
- **Requirements Compliance:** {review_result.get('requirements_compliance', {}).get('summary', 'N/A')}
- **Security & Best Practices:** {review_result.get('security_analysis', {}).get('summary', 'N/A')}

### 💡 Recommendation: **{overall.get('recommendation', 'unknown').upper()}**

{overall.get('summary', 'No summary available')}

---
*This review was generated automatically by AI Coding Agent*
*Debug: SHA {head_sha[:8] if head_sha != 'unknown' else 'unknown'} | Source: {data_source} | Repo: {context.get('repo_full_name', 'unknown')} | PR: #{context.get('pr_number', 'unknown')}*"""
        
        return comment
    
    async def _decide_next_action(
        self,
        iteration: IssueIteration,
        review_result: Dict,
        context: Dict[str, Any] = None
    ) -> None:
        """Decide what to do next based on review results with DB-first CI status."""
        try:
            overall = review_result.get("overall_assessment", {})
            recommendation = overall.get("recommendation", "")
            
            # DB-first CI status logic: prioritize webhook data over API data
            ci_conclusion = None
            ci_data_source = "unknown"
            
            # First, check if we have recent webhook data for the current PR head SHA
            if iteration.pr_head_sha and iteration.ci_status_source == "webhook":
                # Check if webhook data is recent (within 5 minutes)
                from datetime import datetime, timedelta
                if (iteration.ci_status_updated_at and
                    datetime.utcnow() - iteration.ci_status_updated_at < timedelta(minutes=5)):
                    ci_conclusion = iteration.last_ci_conclusion
                    ci_data_source = "webhook (recent)"
                    logger.info(f"Using recent webhook CI data for SHA {iteration.pr_head_sha[:8] if iteration.pr_head_sha else 'unknown'}: {ci_conclusion}")
                else:
                    logger.info(f"Webhook CI data is stale ({iteration.ci_status_updated_at}), will use API fallback")
            
            # Fallback to API data if no recent webhook data
            if not ci_conclusion or ci_conclusion == "unknown":
                try:
                    owner, repo = iteration.repo_full_name.split("/")
                    async with await get_github_client(iteration.installation_id) as github:
                        # Get current PR head SHA for validation
                        pr_data = await github.get_pull_request(owner, repo, iteration.pr_number)
                        current_head_sha = pr_data.get("head", {}).get("sha")
                        
                        if current_head_sha:
                            logger.info(f"Getting API CI status for current PR head SHA {current_head_sha[:8]}")
                            ci_details = await github.get_pr_ci_status(
                                owner, repo, iteration.pr_number, expected_sha=current_head_sha
                            )
                            
                            if ci_details.get("overall_conclusion") and ci_details["overall_conclusion"] != "unknown":
                                ci_conclusion = ci_details["overall_conclusion"]
                                ci_data_source = "api (fallback)"
                                logger.info(f"Using API CI data for SHA {current_head_sha[:8]}: {ci_conclusion}")
                                
                                # Update database with API data if it's more recent
                                if not iteration.pr_head_sha or iteration.pr_head_sha != current_head_sha:
                                    logger.info(f"Updating iteration with current PR head SHA {current_head_sha[:8]}")
                                    db_manager.update_ci_status(
                                        iteration.id,
                                        ci_status=ci_details.get("overall_status", "unknown"),
                                        ci_conclusion=ci_conclusion,
                                        pr_head_sha=current_head_sha,
                                        source="api"
                                    )
                            else:
                                logger.warning(f"API returned unknown CI status for SHA {current_head_sha[:8]}")
                        else:
                            logger.error("Could not get current PR head SHA")
                            
                except Exception as e:
                    logger.error(f"Failed to get API CI status: {e}")
            
            # Final fallback to stored iteration data
            if not ci_conclusion or ci_conclusion == "unknown":
                ci_conclusion = iteration.last_ci_conclusion or "unknown"
                ci_data_source = "stored (fallback)"
                logger.info(f"Using stored CI data: {ci_conclusion}")
            
            ci_success = ci_conclusion == "success"
            logger.info(f"Final CI decision for iteration {iteration.id}: {ci_conclusion} (source: {ci_data_source})")
            
            # Complete if approved AND CI passed - STRICT CI REQUIREMENT
            if recommendation in ["approve", "approve_with_suggestions"]:
                if ci_success:
                    await self._complete_iteration(
                        iteration,
                        IterationStatus.COMPLETED,
                        f"Code approved ({recommendation}) and CI passed (source: {ci_data_source})"
                    )
                else:
                    # CI failed - continue iteration or fail, NEVER complete as successful
                    if iteration.current_iteration < iteration.max_iterations:
                        logger.info(f"Code approved but CI failed ({ci_conclusion}), starting new iteration to fix CI issues")
                        await self._start_new_iteration_for_ci_failure(iteration, ci_conclusion, ci_data_source)
                    else:
                        await self._complete_iteration(
                            iteration,
                            IterationStatus.FAILED,
                            f"Code approved ({recommendation}) but CI failed: {ci_conclusion} (source: {ci_data_source}). Max iterations reached."
                        )
                return
            
            # Continue iteration if changes requested and under limit
            if (recommendation in ["request_changes", "reject"] and
                iteration.current_iteration < iteration.max_iterations):
                
                logger.info(f"Review requested changes, starting iteration {iteration.current_iteration + 1}")
                
                # Update status to trigger next iteration and store feedback
                feedback_text = review_result.get("overall_assessment", {}).get("summary", "")
                if not feedback_text:
                    # Collect feedback from different sections
                    feedback_parts = []
                    if review_result.get("code_quality", {}).get("issues"):
                        feedback_parts.append("Code Quality Issues: " + "; ".join(review_result["code_quality"]["issues"]))
                    if review_result.get("requirements_compliance", {}).get("missing_requirements"):
                        feedback_parts.append("Missing Requirements: " + "; ".join(review_result["requirements_compliance"]["missing_requirements"]))
                    if review_result.get("security_analysis", {}).get("issues"):
                        feedback_parts.append("Security Issues: " + "; ".join(review_result["security_analysis"]["issues"]))
                    feedback_text = ". ".join(feedback_parts) if feedback_parts else "Please address the review comments."
                
                db_manager.update_iteration(
                    iteration.id,
                    last_review_feedback=feedback_text,
                    status=IterationStatus.RUNNING
                )
                
                # Schedule next iteration
                asyncio.create_task(self._run_code_iteration(iteration))
                return
            
            # Fail if max iterations reached
            if iteration.current_iteration >= iteration.max_iterations:
                await self._complete_iteration(
                    iteration,
                    IterationStatus.FAILED,
                    f"Maximum iterations ({iteration.max_iterations}) reached. Last recommendation: {recommendation}, CI: {ci_conclusion}"
                )
            else:
                # Handle unhandled cases
                await self._complete_iteration(
                    iteration,
                    IterationStatus.FAILED,
                    f"Unhandled case: recommendation={recommendation}, CI={ci_conclusion} (source: {ci_data_source})"
                )
            
        except Exception as e:
            logger.error(f"Failed to decide next action: {e}", exc_info=True)
            await self._complete_iteration(iteration, IterationStatus.FAILED, str(e))

    async def _start_new_iteration_for_ci_failure(
        self,
        iteration: IssueIteration,
        ci_conclusion: str,
        ci_data_source: str
    ) -> None:
        """Start new iteration specifically to fix CI failures."""
        try:
            # Create feedback message about CI failure
            ci_feedback = f"CI failed with status: {ci_conclusion} (source: {ci_data_source}). Please fix the issues that caused CI to fail and ensure all tests pass."
            
            # Update iteration with CI failure feedback
            db_manager.update_iteration(
                iteration.id,
                last_review_feedback=ci_feedback,
                status=IterationStatus.RUNNING
            )
            
            # Post comment about starting new iteration for CI fix
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    ci_fix_comment = f"""## 🔄 Starting New Iteration - CI Failed

The code was approved but CI failed with status: **{ci_conclusion}**

**Data Source:** {ci_data_source}
**Action:** Starting iteration {iteration.current_iteration + 1} to fix CI issues

The agent will analyze the CI failure and attempt to fix the issues that caused the build to fail."""
                    
                    await github.create_issue_comment(
                        owner, repo, iteration.pr_number, ci_fix_comment
                    )
            
            # Schedule next iteration to fix CI issues
            asyncio.create_task(self._run_code_iteration(iteration))
            
            logger.info(f"Started new iteration for CI failure fix: {ci_conclusion}")
            
        except Exception as e:
            logger.error(f"Failed to start new iteration for CI failure: {e}", exc_info=True)
            # Fallback to completing as failed
            await self._complete_iteration(
                iteration,
                IterationStatus.FAILED,
                f"Code approved but CI failed: {ci_conclusion}. Failed to start new iteration: {str(e)}"
            )

    async def _complete_iteration(
        self,
        iteration: IssueIteration,
        status: IterationStatus,
        message: str
    ) -> None:
        """Complete iteration with final status."""
        try:
            db_manager.complete_iteration(iteration.id, status)
            
            # Post final comment to PR if exists
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    if status == IterationStatus.COMPLETED:
                        final_comment = f"""## ✅ SDLC Cycle Completed Successfully!

The automated development cycle has been completed successfully after {iteration.current_iteration} iteration(s).

**Final Status:** {message}

This PR is ready for human review and merge."""
                    else:
                        final_comment = f"""## ❌ SDLC Cycle Failed

The automated development cycle could not be completed successfully.

**Reason:** {message}
**Iterations:** {iteration.current_iteration}/{iteration.max_iterations}

Manual intervention may be required to resolve the remaining issues."""
                    
                    await github.create_issue_comment(
                        owner, repo, iteration.pr_number, final_comment
                    )
            
            logger.info(f"Completed iteration {iteration.id} with status {status}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to complete iteration: {e}", exc_info=True)


# Global orchestrator instance
orchestrator = SDLCOrchestrator()