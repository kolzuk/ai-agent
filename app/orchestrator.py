"""Orchestrator for managing iterative SDLC cycles."""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .database import db_manager, IssueIteration, IterationStatus
from .github_client import get_github_client
from .github_app.auth import github_app_auth
from .config import settings
from .repository_analyzer import RepositoryAnalyzer
from .code_modifier import CodeModifier
from .llm_client import LLMClient
from .reviewer import ReviewerAgent

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
            
            # Lock iteration before starting first iteration
            if not db_manager.lock_iteration(iteration.id):
                logger.error(f"Failed to lock iteration {iteration.id} for first run")
                return None
            
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
            
            # Lock iteration before starting first iteration
            if not db_manager.lock_iteration(iteration.id):
                logger.error(f"Failed to lock iteration {iteration.id} for restart")
                return None
            
            # Start first iteration
            await self._run_code_iteration(iteration)
            
            return iteration
            
        except Exception as e:
            logger.error(f"Failed to restart issue cycle: {e}", exc_info=True)
            return None
    
    async def _run_code_iteration(self, iteration: IssueIteration) -> bool:
        """Run a code generation iteration with proper locking."""
        try:
            # Verify we have the lock
            if not db_manager.is_iteration_locked(iteration.id):
                logger.warning(f"Iteration {iteration.id} is not locked, this should not happen")
                return False
            
            logger.info(f"Running code iteration {iteration.current_iteration + 1} for {iteration.repo_full_name}#{iteration.issue_number}")
            
            # Increment iteration counter with race condition protection
            try:
                iteration = db_manager.increment_iteration(iteration.id)
                if not iteration:
                    logger.error("Failed to increment iteration")
                    db_manager.unlock_iteration(iteration.id)
                    return False
            except Exception as e:
                if "already incremented" in str(e).lower():
                    logger.warning("Iteration already incremented by another process")
                    db_manager.unlock_iteration(iteration.id)
                    return False
                raise
            
            # Check if we've exceeded max iterations
            if iteration.current_iteration >= iteration.max_iterations:
                db_manager.unlock_iteration(iteration.id)
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
                db_manager.unlock_iteration(iteration.id)
                await self._complete_iteration(iteration, IterationStatus.FAILED,
                                             "Code generation failed")
                return False
            
            # Handle case where no PR was created (no changes)
            if result.get("no_changes"):
                db_manager.unlock_iteration(iteration.id)
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
            
            # Unlock iteration after successful completion
            db_manager.unlock_iteration(iteration.id)
            
            if result.get("pr_number"):
                logger.info(f"Code iteration completed, waiting for CI. PR: {result.get('pr_number')}")
            else:
                logger.info(f"Code iteration completed with no PR created (no changes needed)")
                await self._complete_iteration(iteration, IterationStatus.COMPLETED,
                                             "Code changes applied but no PR needed")
            return True
            
        except Exception as e:
            logger.error(f"Code iteration failed: {e}", exc_info=True)
            db_manager.unlock_iteration(iteration.id)
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
                # Update existing PR - this is a subsequent iteration
                pr_title = f"Fix #{issue_number}: {context['issue_title']} (Iteration {context['iteration']})"
                pr_body = self._generate_pr_description(context, analysis)
                
                try:
                    await github.update_pull_request(
                        owner, repo, pr_number, title=pr_title, body=pr_body
                    )
                    logger.info(f"Updated existing PR #{pr_number} for iteration {context['iteration']}")
                    
                    # For existing PRs, always return the PR number - don't check for "no changes"
                    # because we're continuing an existing iteration cycle
                    return {
                        "branch_name": branch_name,
                        "pr_number": pr_number,
                        "analysis": analysis
                    }
                except Exception as e:
                    logger.error(f"Failed to update PR #{pr_number}: {e}")
                    # Continue with the existing PR number anyway
                    return {
                        "branch_name": branch_name,
                        "pr_number": pr_number,
                        "analysis": analysis
                    }
            else:
                # No existing PR - this is the first iteration, create new PR
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
                
                # НОВЫЙ КОД: Поиск целевых файлов для сущностей из issue
                target_files_context = await self._find_and_load_target_files(
                    github, owner, repo, repository_analyzer, context
                )
                
                # Build architecture-aware system prompt
                target_files_section = ""
                if target_files_context:
                    target_files_section = f"""
🎯 EXISTING TARGET FILES FOUND:
{target_files_context}

⚠️ CRITICAL INSTRUCTION: These files already exist and MUST be modified, NOT created.
⚠️ DO NOT create new files with similar names in different languages.
⚠️ USE EXACT FILE PATHS from the list above in your files_to_modify array.
"""
                
                system_prompt = f"""You are an expert software developer analyzing GitHub issues with FULL REPOSITORY CONTEXT.

EXISTING REPOSITORY STRUCTURE:
{repository_map.get_structure_summary()}

RELEVANT EXISTING CLASSES:
{self._format_relevant_classes(relevant_classes)}

SUGGESTED MODIFICATION TARGETS:
{self._format_modification_targets(modification_targets)}

ARCHITECTURE PATTERNS DETECTED:
{', '.join(repository_map.architecture_patterns) if repository_map.architecture_patterns else 'Standard Python project'}
{target_files_section}
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

    async def _find_and_load_target_files(self, github, owner: str, repo: str, repository_analyzer: RepositoryAnalyzer, context: Dict[str, Any]) -> str:
        """Найти и загрузить содержимое целевых файлов для issue."""
        try:
            # Извлечь потенциальные имена сущностей из issue
            issue_text = f"{context['issue_title']} {context['issue_body'] or ''}"
            entity_names = self._extract_entity_names_from_issue(issue_text)
            
            target_files_info = []
            
            for entity_name in entity_names:
                logger.info(f"Searching for target files for entity: {entity_name}")
                
                # Найти кандидатов
                candidates = await repository_analyzer.find_target_files_for_entity(
                    github, owner, repo, entity_name
                )
                
                # Выбрать лучший файл
                target_file = repository_analyzer.select_best_target_file(candidates, entity_name)
                
                if target_file:
                    # Загрузить содержимое файла
                    file_content = await self._load_file_content_for_context(
                        github, owner, repo, target_file
                    )
                    
                    if file_content:
                        target_files_info.append(f"""
TARGET FILE: {target_file}
ENTITY: {entity_name}
CONTENT PREVIEW:
```
{file_content[:1000]}{'...' if len(file_content) > 1000 else ''}
```
""")
                        logger.info(f"Loaded target file content: {target_file}")
            
            return "\n".join(target_files_info) if target_files_info else ""
            
        except Exception as e:
            logger.error(f"Failed to find and load target files: {e}")
            return ""

    def _extract_entity_names_from_issue(self, issue_text: str) -> List[str]:
        """Извлечь имена сущностей из текста issue."""
        import re
        
        # Паттерны для поиска имён классов/объектов
        patterns = [
            r'\b([A-Z][a-zA-Z0-9]*(?:Generator|Manager|Service|Controller|Handler|Processor))\b',
            r'\b([A-Z][a-zA-Z0-9]*)\s+(?:class|object|trait)',
            r'`([A-Z][a-zA-Z0-9]*)`',  # Имена в backticks
            r'"([A-Z][a-zA-Z0-9]*)"',  # Имена в кавычках
        ]
        
        entity_names = set()
        for pattern in patterns:
            matches = re.findall(pattern, issue_text)
            entity_names.update(matches)
        
        # Фильтровать слишком короткие или общие имена
        filtered_names = [name for name in entity_names if len(name) > 2 and name not in ['String', 'Int', 'Boolean']]
        
        logger.info(f"Extracted entity names from issue: {filtered_names}")
        return filtered_names

    async def _load_file_content_for_context(self, github, owner: str, repo: str, file_path: str) -> Optional[str]:
        """Загрузить содержимое файла для контекста."""
        try:
            file_data = await github.get_file_content(owner, repo, file_path, "main")
            if file_data:
                import base64
                content = base64.b64decode(file_data["content"]).decode('utf-8')
                return content
        except Exception as e:
            logger.warning(f"Could not load content for {file_path}: {e}")
        return None
    
    def _format_relevant_classes(self, relevant_classes) -> str:
        """Format relevant classes for LLM prompt."""
        if not relevant_classes:
            return "No directly relevant classes found."
        
        formatted = []
        for cls in relevant_classes[:10]:  # Limit to top 10 to avoid token limits
            methods_str = ", ".join([method.name for method in cls.methods[:5]])  # Show first 5 methods
            if len(cls.methods) > 5:
                methods_str += f" (and {len(cls.methods) - 5} more)"
            
            formatted.append(f"- {cls.name} in {cls.file_path}")
            if cls.purpose:
                formatted.append(f"  Purpose: {cls.purpose}")
            formatted.append(f"  Methods: {methods_str}")
            # Убрать строку с dependencies, так как этого поля нет в ClassInfo
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_modification_targets(self, modification_targets) -> str:
        """Format modification targets for LLM prompt with safe attribute access."""
        if not modification_targets:
            return "No specific modification targets identified."
        
        formatted = []
        for target in modification_targets[:5]:  # Limit to top 5
            try:
                # Безопасный доступ к атрибутам с fallback значениями
                target_type = getattr(target, 'target_type', 'Unknown')
                name = getattr(target, 'name', 'Unknown')
                file_path = getattr(target, 'file_path', 'Unknown')
                reason = getattr(target, 'reason', 'No reason provided')
                confidence = getattr(target, 'confidence', 'Unknown')
                
                formatted.append(f"- {target_type}: {name} in {file_path}")
                formatted.append(f"  Reason: {reason}")
                formatted.append(f"  Confidence: {confidence}")
                formatted.append("")
                
            except Exception as e:
                # Fallback для случаев, когда target не имеет ожидаемой структуры
                logger.warning(f"Failed to format modification target: {e}")
                if hasattr(target, 'name'):
                    formatted.append(f"- Target: {target.name}")
                elif hasattr(target, '__str__'):
                    formatted.append(f"- Target: {str(target)}")
                else:
                    formatted.append(f"- Target: {type(target).__name__}")
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
            # Теперь repo_files содержит все файлы рекурсивно
            
            # EXISTENCE CHECK: Validate files to create against existing classes
            validated_analysis = await self._validate_file_operations(analysis, context)
            
            # Process files to modify
            for file_path in validated_analysis.get("files_to_modify", []):
                success = await self._modify_file(
                    github, owner, repo, branch_name, file_path, validated_analysis, context
                )
                if not success:
                    logger.warning(f"Failed to modify {file_path}")
            
            # Process files to create (after validation)
            for file_path in validated_analysis.get("files_to_create", []):
                success = await self._create_file(
                    github, owner, repo, branch_name, file_path, validated_analysis, context
                )
                if not success:
                    logger.warning(f"Failed to create {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply code changes: {e}", exc_info=True)
            return False
    
    async def _validate_file_operations(self, analysis: Dict, context: Dict[str, Any]) -> Dict:
        """Validate file operations and enforce existence checks and scope guardrails."""
        try:
            logger.info("Validating file operations with existence checks and scope guardrails")
            
            # Get repository map from analysis
            repository_map = analysis.get("repository_map")
            if not repository_map:
                logger.warning("No repository map available for existence check")
                return analysis
            
            files_to_create = analysis.get("files_to_create", [])
            files_to_modify = analysis.get("files_to_modify", [])
            
            # SCOPE GUARDRAILS: Analyze task type and restrict operations
            task_type = self._analyze_task_type(context)
            scope_restrictions = self._get_scope_restrictions(task_type)
            
            validated_create = []
            validated_modify = list(files_to_modify)  # Start with existing modify list
            existence_check_results = []
            scope_check_results = []
            
            # Check each file to create for existing classes and scope restrictions
            for file_path in files_to_create:
                # SCOPE GUARDRAILS: Check if file creation is allowed for this task type
                if not scope_restrictions.get("allow_file_creation", True):
                    scope_check_results.append({
                        "intended_file": file_path,
                        "action": "block_create",
                        "reason": f"Task type '{task_type}' does not allow file creation",
                        "task_type": task_type
                    })
                    logger.info(f"Blocking creation of {file_path} - task type '{task_type}' only allows modifications")
                    continue
                
                # Extract potential class name from file path
                potential_class_names = self._extract_class_names_from_path(file_path)
                
                existing_classes_found = []
                for class_name in potential_class_names:
                    # Check if class already exists
                    existing_class = repository_map.classes.get(class_name)
                    if existing_class:
                        existing_classes_found.append(existing_class)
                        logger.info(f"Found existing class {class_name} in {existing_class.file_path}")
                    
                    # Also check fuzzy matches
                    for existing_name, existing_class in repository_map.classes.items():
                        if (class_name.lower() in existing_name.lower() or
                            existing_name.lower() in class_name.lower()):
                            if existing_class not in existing_classes_found:
                                existing_classes_found.append(existing_class)
                                logger.info(f"Found similar existing class {existing_name} in {existing_class.file_path}")
                
                if existing_classes_found:
                    # Move existing files to modify list instead of create
                    for existing_class in existing_classes_found:
                        if existing_class.file_path not in validated_modify:
                            validated_modify.append(existing_class.file_path)
                            logger.info(f"Moving {existing_class.file_path} from create to modify list")
                    
                    existence_check_results.append({
                        "intended_file": file_path,
                        "action": "redirect_to_modify",
                        "existing_classes": [cls.name for cls in existing_classes_found],
                        "existing_files": [cls.file_path for cls in existing_classes_found]
                    })
                else:
                    # No existing class found, check if creation is allowed
                    if scope_restrictions.get("allow_file_creation", True):
                        validated_create.append(file_path)
                        existence_check_results.append({
                            "intended_file": file_path,
                            "action": "allow_create",
                            "reason": "No existing class found and task type allows creation"
                        })
                    else:
                        scope_check_results.append({
                            "intended_file": file_path,
                            "action": "block_create",
                            "reason": f"Task type '{task_type}' does not allow new file creation",
                            "task_type": task_type
                        })
            
            # Check for language compatibility
            project_profile = repository_map.project_profile
            if project_profile and project_profile.primary_language:
                language_validated_create = []
                for file_path in validated_create:
                    if self._is_language_compatible(file_path, project_profile):
                        language_validated_create.append(file_path)
                    else:
                        logger.warning(f"Blocking creation of {file_path} - incompatible with {project_profile.primary_language} project")
                        existence_check_results.append({
                            "intended_file": file_path,
                            "action": "block_create",
                            "reason": f"Language incompatible with {project_profile.primary_language} project"
                        })
                validated_create = language_validated_create
            
            # СТРОГАЯ ПРОВЕРКА ЯЗЫКА: дополнительная валидация для операций создания файлов
            if project_profile:
                final_validated_create = []
                for file_path in validated_create:
                    file_ext = file_path.split('.')[-1].lower() if '.' in file_path else "unknown"
                    
                    # Определить язык проекта
                    project_lang = "Unknown"
                    if project_profile.is_scala_project:
                        project_lang = "Scala"
                    elif project_profile.is_java_project:
                        project_lang = "Java"
                    elif project_profile.is_python_project:
                        project_lang = "Python"
                    
                    # Строгая проверка совместимости языка
                    is_compatible = True
                    blocking_reason = ""
                    
                    if project_profile.is_scala_project and file_ext == "java":
                        is_compatible = False
                        blocking_reason = f"Cannot create Java files in Scala project"
                    elif project_profile.is_java_project and file_ext == "scala":
                        is_compatible = False
                        blocking_reason = f"Cannot create Scala files in Java project"
                    elif project_profile.is_python_project and file_ext not in ["py"]:
                        is_compatible = False
                        blocking_reason = f"Cannot create {file_ext.upper()} files in Python project"
                    
                    if is_compatible:
                        final_validated_create.append(file_path)
                        logger.info(f"✅ LANGUAGE VALIDATION PASSED: {file_path} is compatible with {project_lang} project")
                    else:
                        logger.error(f"🚫 STRICT LANGUAGE VALIDATION FAILED: {blocking_reason}")
                        logger.error(f"   File: {file_path} (extension: {file_ext})")
                        logger.error(f"   Project: {project_lang} (Scala={project_profile.is_scala_project}, Java={project_profile.is_java_project}, Python={project_profile.is_python_project})")
                        
                        existence_check_results.append({
                            "intended_file": file_path,
                            "action": "block_create",
                            "reason": f"STRICT LANGUAGE VALIDATION: {blocking_reason}"
                        })
                
                validated_create = final_validated_create
            
            # Update analysis with validated file lists
            validated_analysis = analysis.copy()
            validated_analysis["files_to_create"] = validated_create
            validated_analysis["files_to_modify"] = list(set(validated_modify))  # Remove duplicates
            validated_analysis["existence_check_results"] = existence_check_results
            validated_analysis["scope_check_results"] = scope_check_results
            validated_analysis["task_type"] = task_type
            validated_analysis["scope_restrictions"] = scope_restrictions
            
            # Log validation results
            logger.info(f"Validation results for task type '{task_type}':")
            logger.info(f"  - Original files to create: {len(files_to_create)}")
            logger.info(f"  - Validated files to create: {len(validated_create)}")
            logger.info(f"  - Files redirected to modify: {len(validated_modify) - len(files_to_modify)}")
            logger.info(f"  - Files blocked by scope: {len(scope_check_results)}")
            
            for result in existence_check_results:
                logger.info(f"  - {result['intended_file']}: {result['action']} - {result.get('reason', 'See existing_classes')}")
            
            for result in scope_check_results:
                logger.info(f"  - {result['intended_file']}: {result['action']} - {result['reason']}")
            
            return validated_analysis
            
        except Exception as e:
            logger.error(f"Failed to validate file operations: {e}", exc_info=True)
            return analysis  # Return original analysis on error
    
    def _extract_class_names_from_path(self, file_path: str) -> List[str]:
        """Extract potential class names from file path."""
        try:
            # Get filename without extension
            filename = file_path.split('/')[-1]
            name_without_ext = filename.split('.')[0]
            
            # Convert different naming conventions to class names
            potential_names = []
            
            # Direct name
            potential_names.append(name_without_ext)
            
            # PascalCase conversion
            if '_' in name_without_ext:
                pascal_case = ''.join(word.capitalize() for word in name_without_ext.split('_'))
                potential_names.append(pascal_case)
            
            # CamelCase to PascalCase
            if name_without_ext and name_without_ext[0].islower():
                potential_names.append(name_without_ext.capitalize())
            
            return potential_names
            
        except Exception as e:
            logger.warning(f"Failed to extract class names from {file_path}: {e}")
            return []
    
    def _is_language_compatible(self, file_path: str, project_profile) -> bool:
        """Check if file language is compatible with project."""
        try:
            file_ext = file_path.split('.')[-1].lower()
            primary_lang = project_profile.primary_language.lower()
            
            # Language compatibility rules
            if primary_lang == "scala":
                # Scala projects should not create Java files
                return file_ext != "java"
            elif primary_lang == "java":
                # Java projects should not create Scala files
                return file_ext != "scala"
            elif primary_lang == "python":
                # Python projects should primarily use Python
                return file_ext == "py"
            
            # Default: allow creation
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check language compatibility for {file_path}: {e}")
            return True
    
    def _analyze_task_type(self, context: Dict[str, Any]) -> str:
        """Analyze the task type based on issue title and body."""
        try:
            issue_title = context.get("issue_title", "").lower()
            issue_body = context.get("issue_body", "").lower()
            combined_text = f"{issue_title} {issue_body}"
            
            # Comment-related tasks
            if any(keyword in combined_text for keyword in [
                "add comment", "добавить комментар", "comment", "комментар",
                "document", "документ", "javadoc", "scaladoc"
            ]):
                return "add_comments"
            
            # Bug fix tasks
            if any(keyword in combined_text for keyword in [
                "fix bug", "исправить баг", "bug", "баг", "error", "ошибка",
                "issue", "проблема", "broken", "сломан"
            ]):
                return "fix_bug"
            
            # Refactoring tasks
            if any(keyword in combined_text for keyword in [
                "refactor", "рефактор", "cleanup", "clean up", "optimize",
                "improve", "улучшить", "restructure"
            ]):
                return "refactor"
            
            # New feature tasks
            if any(keyword in combined_text for keyword in [
                "add feature", "добавить функци", "new feature", "новая функци",
                "implement", "реализовать", "create", "создать"
            ]):
                return "add_feature"
            
            # Test-related tasks
            if any(keyword in combined_text for keyword in [
                "test", "тест", "unit test", "integration test"
            ]):
                return "add_tests"
            
            # Default to general task
            return "general"
            
        except Exception as e:
            logger.warning(f"Failed to analyze task type: {e}")
            return "general"
    
    def _get_scope_restrictions(self, task_type: str) -> Dict[str, Any]:
        """Get scope restrictions for a given task type."""
        restrictions = {
            "add_comments": {
                "allow_file_creation": False,
                "allowed_operations": ["modify_existing"],
                "description": "Comment tasks should only modify existing files",
                "max_files_to_modify": 5
            },
            "fix_bug": {
                "allow_file_creation": False,
                "allowed_operations": ["modify_existing"],
                "description": "Bug fixes should only modify existing files",
                "max_files_to_modify": 3
            },
            "refactor": {
                "allow_file_creation": False,
                "allowed_operations": ["modify_existing", "move_files"],
                "description": "Refactoring should primarily modify existing files",
                "max_files_to_modify": 10
            },
            "add_feature": {
                "allow_file_creation": True,
                "allowed_operations": ["modify_existing", "create_new"],
                "description": "Feature additions can create new files",
                "max_files_to_create": 5,
                "max_files_to_modify": 10
            },
            "add_tests": {
                "allow_file_creation": True,
                "allowed_operations": ["create_new", "modify_existing"],
                "description": "Test additions can create new test files",
                "max_files_to_create": 3,
                "max_files_to_modify": 5
            },
            "general": {
                "allow_file_creation": True,
                "allowed_operations": ["modify_existing", "create_new"],
                "description": "General tasks have no specific restrictions",
                "max_files_to_create": 3,
                "max_files_to_modify": 5
            }
        }
        
        return restrictions.get(task_type, restrictions["general"])
    
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
            logger.info(f"=== ATTEMPTING TO MODIFY FILE: {file_path} ===")
            
            # Получить все файлы репозитория для проверки существования
            all_files = await github.list_repository_files(owner, repo)
            existing_files = {file_item["path"] for file_item in all_files}
            
            logger.info(f"Repository contains {len(existing_files)} files")
            logger.info(f"Checking if {file_path} exists in repository...")
            
            if file_path not in existing_files:
                logger.warning(f"❌ FILE NOT FOUND: {file_path}")
                logger.info(f"Available files matching pattern:")
                
                # Поиск похожих файлов для диагностики
                file_name = file_path.split('/')[-1].split('.')[0]
                similar_files = [f for f in existing_files if file_name.lower() in f.lower()]
                for similar in similar_files[:5]:
                    logger.info(f"  - {similar}")
                
                # Проверить ограничения задачи
                task_type = analysis.get("task_type", "general")
                
                if task_type == "add_comments":
                    logger.error(f"File {file_path} not found and task type '{task_type}' does not allow file creation")
                    
                    # Завершить итерацию с сообщением об ошибке
                    await self._complete_iteration_with_file_not_found_error(
                        context, file_path, analysis
                    )
                    return False
                else:
                    logger.info(f"File not found, creating new file: {file_path}")
                    return await self._create_file(github, owner, repo, branch_name, file_path, analysis, context)
            
            # Файл существует, модифицировать его
            logger.info(f"✅ FILE EXISTS: {file_path}, proceeding with modification")
            
            # Get current file content
            file_data = await github.get_file_content(owner, repo, file_path, branch_name)
            if not file_data:
                # Попробовать из main ветки
                file_data = await github.get_file_content(owner, repo, file_path, "main")
            
            if not file_data:
                logger.error(f"Could not retrieve content for {file_path}")
                return False
            
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
                
                # Initialize CodeModifier with repository_map
                code_modifier = CodeModifier(repository_map)
                
                # Find if this file has relevant classes to modify
                file_classes = [cls for cls in relevant_classes if cls.file_path == file_path]
                file_targets = [target for target in modification_targets if target.file_path == file_path]
                
                if file_classes or file_targets:
                    # Use intelligent code modification with existing methods
                    try:
                        # Generate architecture-aware prompt
                        requirement = analysis.get('summary', '') + ' ' + analysis.get('technical_approach', '')
                        arch_prompt = code_modifier.generate_architecture_aware_prompt(requirement, [file_path])
                        
                        # For now, fall back to basic LLM modification with architecture context
                        logger.info(f"Using architecture-aware context for {file_path}")
                        # We'll use the architecture context in the basic modification below
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate architecture context for {file_path}: {e}")
            
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
        """Run review iteration with SHA-aware CI status checking and ANTI-DUPLICATION."""
        try:
            # 🚨 CRITICAL: Check if iteration is already being processed to prevent duplication
            if db_manager.is_iteration_locked(iteration.id):
                logger.warning(f"🔒 REVIEW DUPLICATION BLOCKED: Iteration {iteration.id} is already being processed")
                logger.warning(f"   📋 This prevents duplicate review comments")
                return True
            
            # 🚨 CRITICAL: Lock iteration for review processing
            if not db_manager.lock_iteration(iteration.id):
                logger.warning(f"🔒 FAILED TO LOCK iteration {iteration.id} for review - another process may be handling it")
                return False
            
            logger.critical(f"🎯 STARTING REVIEW for {iteration.repo_full_name}#{iteration.issue_number}")
            logger.critical(f"   🔒 Iteration {iteration.id} LOCKED for review processing")
            logger.critical(f"   📊 Current CI status: {iteration.last_ci_status}, conclusion: {iteration.last_ci_conclusion}")
            logger.critical(f"   🔗 SHA: {iteration.pr_head_sha[:8] if iteration.pr_head_sha else 'none'}")
            
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
                    
                    # Добавить обработку unhandled CI events:
                    if actual_ci_conclusion == "unknown":
                        logger.warning(f"CI status is unknown for PR #{iteration.pr_number} - this will block approval")
                        
                        # Проверить, есть ли unhandled CI events в логах
                        # (это можно сделать через анализ recent webhook events или CI API)
                        unhandled_events_detected = await self._check_for_unhandled_ci_events(
                            github, owner, repo, iteration.pr_number
                        )
                        
                        if unhandled_events_detected:
                            logger.warning(f"Detected unhandled CI events for PR #{iteration.pr_number}")
                            actual_ci_conclusion = "unknown_unhandled"
                            ci_data_source += " (unhandled events detected)"
                    
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
                
                # Decide next action (may continue iteration with same lock)
                continue_iteration = await self._decide_next_action(iteration, review_result, review_context)
                
                # 🚨 CRITICAL: Only unlock if iteration is NOT continuing
                if not continue_iteration:
                    db_manager.unlock_iteration(iteration.id)
                    logger.critical(f"🔓 REVIEW COMPLETED - Iteration {iteration.id} UNLOCKED (no continuation)")
                else:
                    logger.critical(f"🔒 REVIEW COMPLETED - Iteration {iteration.id} LOCK MAINTAINED for continuation")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Review iteration failed: {e}", exc_info=True)
            # 🚨 CRITICAL: Always unlock on error
            db_manager.unlock_iteration(iteration.id)
            logger.critical(f"🔓 ERROR CLEANUP - Iteration {iteration.id} UNLOCKED")
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
                from .github_client import GitHubAppClient as GitHubClient
                
                # Create GitHub client using installation_id (our new unified approach)
                github_client = GitHubClient(context.get("installation_id"))
                
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
                
                # Perform comprehensive review with architecture context and CI gate
                review_result = await reviewer_agent._perform_comprehensive_review(
                    context["pr_data"],
                    issue_data,
                    pr_files_data,
                    context.get("ci_conclusion")
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
        """Post review results to PR with STRICT CI GATE enforcement."""
        try:
            owner, repo = context["repo_full_name"].split("/")
            
            # STRICT CI GATE: Apply CI gate to review result BEFORE formatting comment
            ci_conclusion = context.get("ci_conclusion", "unknown")
            original_recommendation = review_result.get("overall_assessment", {}).get("recommendation", "")
            
            # Force recommendation and status when CI is not success
            if ci_conclusion not in ["success"]:
                # Override the review result to enforce STRICT CI GATE
                if ci_conclusion == "unknown":
                    blocking_reason = "CI status is unknown - cannot approve without confirmed successful CI"
                    logger.info(f"CI status unknown, forcing REQUEST_CHANGES regardless of original recommendation ({original_recommendation})")
                else:
                    blocking_reason = f"CI failed with status: {ci_conclusion}"
                    logger.info(f"CI failed ({ci_conclusion}), forcing REQUEST_CHANGES regardless of original recommendation ({original_recommendation})")
                
                # Override the overall assessment to reflect CI blocking
                review_result["overall_assessment"]["recommendation"] = "request_changes"
                review_result["overall_assessment"]["status"] = "🔄 Changes requested (CI failing)"
                
                # Add blocking reason to summary
                original_summary = review_result.get("overall_assessment", {}).get("summary", "")
                review_result["overall_assessment"]["summary"] = f"{original_summary}\n\n🚫 **BLOCKING**: {blocking_reason}"
                
                # Apply penalty to score when CI fails
                original_score = review_result.get("overall_assessment", {}).get("score", 0)
                if original_score > 50:
                    review_result["overall_assessment"]["score"] = max(30, original_score * 0.6)  # Significant penalty
                
                event = "REQUEST_CHANGES"
            elif original_recommendation == "approve" and ci_conclusion == "success":
                # 🚨 CRITICAL: NEVER use APPROVE to prevent automatic PR closure
                event = "COMMENT"
                logger.critical(f"🚫 BLOCKED APPROVE EVENT: Using COMMENT instead to prevent PR auto-closure")
                logger.critical(f"   📊 Would have been APPROVE but blocked for safety")
            elif original_recommendation == "request_changes":
                event = "REQUEST_CHANGES"
            else:
                event = "COMMENT"
            
            # Generate review comment with updated review result
            comment = self._format_review_comment(review_result, context)
            
            # 🚨 CRITICAL FIX: Create ONLY PR review to prevent duplication
            try:
                await github.create_pull_request_review(
                    owner, repo, context["pr_number"], comment, event
                )
                logger.info(f"✅ Posted SINGLE review to PR #{context['pr_number']} with event: {event}")
                logger.info(f"🔧 ANTI-DUPLICATION: Only PR review created, no separate comment")
            except Exception as review_e:
                error_str = str(review_e).lower()
                if "422" in error_str:
                    logger.warning(f"❌ Failed to create PR review: {review_e}")
                    logger.info(f"🔄 FALLBACK: Creating issue comment instead")
                    # Fallback to comment only if PR review fails
                    await github.create_issue_comment(
                        owner, repo, context["pr_number"], comment
                    )
                    logger.info(f"✅ Posted fallback comment to PR #{context['pr_number']}")
                else:
                    logger.error(f"❌ Unexpected error creating PR review: {review_e}")
                    raise review_e
            
            logger.info(f"Posted review results to PR #{context['pr_number']} with event: {event}")
            
        except Exception as e:
            logger.error(f"Failed to post review results: {e}", exc_info=True)
    
    def _format_review_comment(self, review_result: Dict, context: Dict[str, Any]) -> str:
        """Format review comment with STRICT CI GATE enforcement."""
        overall = review_result.get("overall_assessment", {})
        
        # Get SHA and data source info for enhanced logging
        ci_details = context.get('ci_status_details', {})
        head_sha = ci_details.get('head_sha', 'unknown')
        data_source = context.get('ci_data_source', 'unknown')
        ci_conclusion = context.get('ci_conclusion', 'unknown')
        
        # Determine the display status based on CI gate enforcement
        display_status = overall.get('status', 'Review Completed')
        
        comment = f"""## 🤖 AI Code Review - Iteration {context['iteration']}

### {display_status}

**Overall Score:** {overall.get('score', 0)}/100

**CI Status:** {ci_conclusion} ({'✅' if ci_conclusion == 'success' else '❌'})
**CI Data Source:** {data_source}
**Commit SHA:** `{head_sha[:8] if head_sha != 'unknown' else 'unknown'}`

**Detailed CI Status:**"""

        # Add detailed CI information if available
        if ci_details and ci_details.get('overall_conclusion') and ci_details.get('overall_conclusion') != 'unknown':
            actual_ci_status = ci_details['overall_conclusion']
            comment = comment.replace(
                f"**CI Status:** {ci_conclusion}",
                f"**CI Status:** {actual_ci_status}"
            )
            comment = comment.replace(
                f"({'✅' if ci_conclusion == 'success' else '❌'})",
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
    ) -> bool:
        """Decide next action with STRICT CI GATE but CONTINUE in same PR.
        
        Returns:
            bool: True if iteration continues (lock should be maintained), False if completed
        """
        try:
            overall = review_result.get("overall_assessment", {})
            recommendation = overall.get("recommendation", "")
            
            # 🚨 CRITICAL: Get FRESH CI conclusion with strict validation
            ci_conclusion = await self._get_ci_conclusion_strict(iteration, context)
            ci_data_source = self._get_ci_data_source(iteration, context)
            
            # 🔍 DETAILED LOGGING for debugging
            logger.critical(f"🎯 DECISION ANALYSIS for iteration {iteration.id}:")
            logger.critical(f"   📊 Recommendation: {recommendation}")
            logger.critical(f"   🚦 CI Conclusion: {ci_conclusion} (source: {ci_data_source})")
            logger.critical(f"   📋 Context CI: {context.get('ci_conclusion') if context else 'no context'}")
            logger.critical(f"   💾 DB CI: {iteration.last_ci_conclusion}")
            logger.critical(f"   🔄 Iteration: {iteration.current_iteration}/{iteration.max_iterations}")
            
            # 🎯 ПРАВИЛО 1: Код хороший + CI зеленый = ЗАВЕРШИТЬ УСПЕШНО
            if recommendation in ["approve", "approve_with_suggestions"] and ci_conclusion == "success":
                logger.critical(f"✅ RULE 1 TRIGGERED: Good code + Green CI")
                logger.critical(f"   📊 Recommendation: {recommendation}")
                logger.critical(f"   🚦 CI Status: {ci_conclusion}")
                
                # 🚨 TRIPLE CHECK: Ensure CI is REALLY success
                if context and context.get('ci_conclusion') and context.get('ci_conclusion') != 'success':
                    logger.error(f"🚨 CRITICAL CONFLICT DETECTED:")
                    logger.error(f"   🎯 Final CI: {ci_conclusion}")
                    logger.error(f"   📋 Context CI: {context.get('ci_conclusion')}")
                    logger.error(f"   🔄 FORCING CI FAILURE PATH instead of success")
                    
                    # Force CI failure path
                    if iteration.current_iteration < iteration.max_iterations:
                        await self._continue_iteration_for_ci_failure(iteration, context.get('ci_conclusion', 'failure'), ci_data_source)
                        return True  # Iteration continues, maintain lock
                    else:
                        await self._complete_iteration(
                            iteration,
                            IterationStatus.FAILED,
                            f"Max iterations reached. Code approved ({recommendation}) but CI conflict detected: final={ci_conclusion}, context={context.get('ci_conclusion')} (source: {ci_data_source})"
                        )
                        return False  # Iteration completed, no continuation
                
                # All checks passed - complete successfully
                logger.critical(f"🎉 COMPLETING ITERATION as SUCCESS")
                await self._complete_iteration(
                    iteration,
                    IterationStatus.COMPLETED,
                    f"Code approved ({recommendation}) and CI passed (source: {ci_data_source})"
                )
                return False  # Iteration completed, no continuation
            
            # 🎯 ПРАВИЛО 2: Код хороший + CI красный = ПРОДОЛЖИТЬ В ТОМ ЖЕ PR
            if recommendation in ["approve", "approve_with_suggestions"] and ci_conclusion != "success":
                logger.critical(f"🔄 RULE 2 TRIGGERED: Good code + Red CI")
                logger.critical(f"   📊 Recommendation: {recommendation}")
                logger.critical(f"   🚦 CI Status: {ci_conclusion}")
                logger.critical(f"   🔄 Current iteration: {iteration.current_iteration}/{iteration.max_iterations}")
                
                if iteration.current_iteration < iteration.max_iterations:
                    logger.critical(f"🔄 CONTINUING iteration in same PR to fix CI")
                    await self._continue_iteration_for_ci_failure(iteration, ci_conclusion, ci_data_source)
                    return True  # Iteration continues, maintain lock
                else:
                    logger.critical(f"❌ MAX ITERATIONS REACHED - completing as FAILED")
                    await self._complete_iteration(
                        iteration,
                        IterationStatus.FAILED,
                        f"Max iterations reached. Code approved ({recommendation}) but CI still failing: {ci_conclusion} (source: {ci_data_source})"
                    )
                    return False  # Iteration completed, no continuation
            
            # 🎯 ПРАВИЛО 3: Код плохой = ПРОДОЛЖИТЬ В ТОМ ЖЕ PR (независимо от CI)
            if recommendation in ["request_changes", "reject"]:
                logger.critical(f"🔄 RULE 3 TRIGGERED: Bad code (CI irrelevant)")
                logger.critical(f"   📊 Recommendation: {recommendation}")
                logger.critical(f"   🚦 CI Status: {ci_conclusion} (ignored for bad code)")
                logger.critical(f"   🔄 Current iteration: {iteration.current_iteration}/{iteration.max_iterations}")
                
                if iteration.current_iteration < iteration.max_iterations:
                    logger.critical(f"🔄 CONTINUING iteration in same PR to fix code issues")
                    await self._continue_iteration_for_code_issues(iteration, review_result)
                    return True  # Iteration continues, maintain lock
                else:
                    logger.critical(f"❌ MAX ITERATIONS REACHED - completing as FAILED")
                    await self._complete_iteration(
                        iteration,
                        IterationStatus.FAILED,
                        f"Max iterations reached. Code issues remain. Last recommendation: {recommendation}, CI: {ci_conclusion}"
                    )
                    return False  # Iteration completed, no continuation
            
            # 🚨 UNHANDLED CASE
            logger.error(f"🚨 UNHANDLED DECISION CASE:")
            logger.error(f"   📊 Recommendation: {recommendation}")
            logger.error(f"   🚦 CI Status: {ci_conclusion}")
            logger.error(f"   🔄 Iteration: {iteration.current_iteration}/{iteration.max_iterations}")
            logger.error(f"   📋 Context: {context}")
            
            await self._complete_iteration(
                iteration,
                IterationStatus.FAILED,
                f"Unhandled decision case: recommendation={recommendation}, CI={ci_conclusion} (source: {ci_data_source})"
            )
            return False  # Iteration completed, no continuation
            
        except Exception as e:
            logger.error(f"Failed to decide next action: {e}", exc_info=True)
            await self._complete_iteration(iteration, IterationStatus.FAILED, str(e))
            return False  # Iteration completed due to error, no continuation

    async def _get_ci_conclusion_strict(self, iteration: IssueIteration, context: Dict[str, Any] = None) -> str:
        """Get CI conclusion with STRICT validation and cross-checking."""
        try:
            logger.info(f"🔍 STRICT CI STATUS CHECK for iteration {iteration.id}")
            
            # Source 1: Context from review (most recent)
            context_ci = context.get('ci_conclusion') if context else None
            logger.info(f"   📋 Context CI: {context_ci}")
            
            # Source 2: Database webhook data
            db_ci = iteration.last_ci_conclusion
            db_source = iteration.ci_status_source
            db_updated = iteration.ci_status_updated_at
            logger.info(f"   💾 DB CI: {db_ci} (source: {db_source}, updated: {db_updated})")
            
            # Source 3: Fresh API call
            api_ci = None
            try:
                owner, repo = iteration.repo_full_name.split("/")
                async with await get_github_client(iteration.installation_id) as github:
                    pr_data = await github.get_pull_request(owner, repo, iteration.pr_number)
                    current_head_sha = pr_data.get("head", {}).get("sha")
                    
                    if current_head_sha:
                        ci_details = await github.get_pr_ci_status(
                            owner, repo, iteration.pr_number, expected_sha=current_head_sha
                        )
                        api_ci = ci_details.get("overall_conclusion")
                        logger.info(f"   🌐 API CI: {api_ci} (SHA: {current_head_sha[:8] if current_head_sha else 'unknown'})")
            except Exception as e:
                logger.error(f"   ❌ API CI check failed: {e}")
            
            # STRICT VALIDATION: Cross-check sources
            sources = [
                ("context", context_ci),
                ("database", db_ci),
                ("api", api_ci)
            ]
            
            # Filter out None/unknown values
            valid_sources = [(name, value) for name, value in sources if value and value != "unknown"]
            
            if not valid_sources:
                logger.warning(f"🚨 NO VALID CI DATA FOUND - defaulting to 'unknown'")
                return "unknown"
            
            # Check for conflicts
            unique_values = set(value for _, value in valid_sources)
            if len(unique_values) > 1:
                logger.error(f"🚨 CI STATUS CONFLICT detected:")
                for name, value in valid_sources:
                    logger.error(f"     {name}: {value}")
                
                # Priority: context > api > database
                if context_ci and context_ci != "unknown":
                    logger.warning(f"🎯 Using CONTEXT CI (highest priority): {context_ci}")
                    return context_ci
                elif api_ci and api_ci != "unknown":
                    logger.warning(f"🎯 Using API CI (medium priority): {api_ci}")
                    return api_ci
                else:
                    logger.warning(f"🎯 Using DB CI (lowest priority): {db_ci}")
                    return db_ci or "unknown"
            
            # No conflicts - use the first valid value
            final_ci = valid_sources[0][1]
            logger.info(f"✅ CONSISTENT CI STATUS: {final_ci}")
            return final_ci
            
        except Exception as e:
            logger.error(f"❌ STRICT CI CHECK FAILED: {e}")
            return "unknown"

    async def _get_ci_conclusion(self, iteration: IssueIteration, context: Dict[str, Any] = None) -> str:
        """Get CI conclusion using DB-first logic with API fallback."""
        try:
            # First, check if we have recent webhook data for the current PR head SHA
            if iteration.pr_head_sha and iteration.ci_status_source == "webhook":
                # Check if webhook data is recent (within 5 minutes)
                from datetime import datetime, timedelta
                if (iteration.ci_status_updated_at and
                    datetime.utcnow() - iteration.ci_status_updated_at < timedelta(minutes=5)):
                    logger.info(f"Using recent webhook CI data for SHA {iteration.pr_head_sha[:8] if iteration.pr_head_sha else 'unknown'}: {iteration.last_ci_conclusion}")
                    return iteration.last_ci_conclusion or "unknown"
                else:
                    logger.info(f"Webhook CI data is stale ({iteration.ci_status_updated_at}), will use API fallback")
            
            # Fallback to API data if no recent webhook data
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
                            return ci_conclusion
                        else:
                            logger.warning(f"API returned unknown CI status for SHA {current_head_sha[:8]}")
                    else:
                        logger.error("Could not get current PR head SHA")
                        
            except Exception as e:
                logger.error(f"Failed to get API CI status: {e}")
            
            # Final fallback to stored iteration data
            ci_conclusion = iteration.last_ci_conclusion or "unknown"
            logger.info(f"Using stored CI data: {ci_conclusion}")
            return ci_conclusion
            
        except Exception as e:
            logger.error(f"Failed to get CI conclusion: {e}")
            return "unknown"

    def _get_ci_data_source(self, iteration: IssueIteration, context: Dict[str, Any] = None) -> str:
        """Get CI data source description."""
        if iteration.pr_head_sha and iteration.ci_status_source == "webhook":
            from datetime import datetime, timedelta
            if (iteration.ci_status_updated_at and
                datetime.utcnow() - iteration.ci_status_updated_at < timedelta(minutes=5)):
                return "webhook (recent)"
            else:
                return "api (fallback)"
        return iteration.ci_status_source or "stored (fallback)"

    async def _continue_iteration_for_ci_failure(
        self,
        iteration: IssueIteration,
        ci_conclusion: str,
        ci_data_source: str = "unknown"
    ) -> None:
        """Continue iteration in the same PR to fix CI failures."""
        try:
            # 🚨 CRITICAL: This method is called from _decide_next_action where iteration is already locked
            # We should NOT check for lock or try to lock again - just proceed
            logger.critical(f"🔄 CONTINUING iteration {iteration.id} for CI failure (already locked)")
            
            logger.info(f"Continuing iteration for CI failure in {iteration.repo_full_name}#{iteration.issue_number} PR#{iteration.pr_number}")
            
            # Analyze CI failure details
            ci_failure_analysis = await self._analyze_ci_failure(iteration, ci_conclusion, ci_data_source)
            
            # Preserve context from previous iterations
            iteration_context = await self._build_iteration_context(iteration)
            
            # Create enhanced feedback message with CI failure analysis
            ci_feedback = self._format_ci_failure_feedback(ci_failure_analysis, iteration_context)
            
            # Update iteration with enhanced CI failure feedback - keep RUNNING status
            db_manager.update_iteration(
                iteration.id,
                last_review_feedback=ci_feedback,
                status=IterationStatus.RUNNING  # Continue in same PR
            )
            
            # Post detailed comment about CI failure analysis
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    ci_fix_comment = self._format_ci_failure_comment(
                        ci_failure_analysis, iteration_context, iteration.current_iteration + 1
                    )
                    
                    await github.create_issue_comment(
                        owner, repo, iteration.pr_number, ci_fix_comment
                    )
            
            # 🚨 CRITICAL: DON'T unlock here - pass the lock to the scheduled iteration
            # Schedule next iteration instead of running immediately
            asyncio.create_task(self._schedule_next_iteration(iteration))
            
            logger.critical(f"🔄 Scheduled next iteration for CI failure with {len(ci_failure_analysis.get('potential_fixes', []))} potential fixes identified (lock maintained)")
            
        except Exception as e:
            logger.error(f"Failed to continue iteration for CI failure: {e}", exc_info=True)
            # Unlock iteration on error
            db_manager.unlock_iteration(iteration.id)
            # Fallback to completing as failed
            await self._complete_iteration(
                iteration,
                IterationStatus.FAILED,
                f"Code approved but CI failed: {ci_conclusion}. Failed to continue iteration: {str(e)}"
            )

    async def _continue_iteration_for_code_issues(
        self,
        iteration: IssueIteration,
        review_result: Dict
    ) -> None:
        """Continue iteration in the same PR to fix code issues."""
        try:
            # 🚨 CRITICAL: This method is called from _decide_next_action where iteration is already locked
            # We should NOT check for lock or try to lock again - just proceed
            logger.critical(f"🔄 CONTINUING iteration {iteration.id} for code issues (already locked)")
            
            logger.info(f"Continuing iteration for code issues in {iteration.repo_full_name}#{iteration.issue_number} PR#{iteration.pr_number}")
            
            # Extract feedback from review result
            overall_assessment = review_result.get("overall_assessment", {})
            feedback_text = overall_assessment.get("summary", "")
            
            if not feedback_text:
                # Collect feedback from different sections
                feedback_parts = []
                if review_result.get("code_quality", {}).get("issues"):
                    issues = review_result["code_quality"]["issues"]
                    feedback_parts.append("Code Quality Issues: " + "; ".join([issue.get("message", "") for issue in issues]))
                if review_result.get("requirements_compliance", {}).get("missing_features"):
                    missing = review_result["requirements_compliance"]["missing_features"]
                    feedback_parts.append("Missing Requirements: " + "; ".join(missing))
                if review_result.get("security_analysis", {}).get("security_issues"):
                    sec_issues = review_result["security_analysis"]["security_issues"]
                    feedback_parts.append("Security Issues: " + "; ".join([issue.get("issue", "") for issue in sec_issues]))
                feedback_text = ". ".join(feedback_parts) if feedback_parts else "Please address the review comments."
            
            # Update iteration with code issue feedback - keep RUNNING status
            db_manager.update_iteration(
                iteration.id,
                last_review_feedback=feedback_text,
                status=IterationStatus.RUNNING  # Continue in same PR
            )
            
            # 🚨 CRITICAL: DON'T unlock here - pass the lock to the scheduled iteration
            # Schedule next iteration instead of running immediately
            asyncio.create_task(self._schedule_next_iteration(iteration))
            
            logger.critical(f"🔄 Scheduled next iteration for code issues (lock maintained)")
            
        except Exception as e:
            logger.error(f"Failed to continue iteration for code issues: {e}", exc_info=True)
            # Unlock iteration on error
            db_manager.unlock_iteration(iteration.id)
            # Fallback to completing as failed
            await self._complete_iteration(
                iteration,
                IterationStatus.FAILED,
                f"Failed to continue iteration for code issues: {str(e)}"
            )

    async def _schedule_next_iteration(self, iteration: IssueIteration, delay_seconds: int = 5) -> None:
        """Schedule next iteration with delay to avoid race conditions."""
        try:
            # Delay to avoid race conditions
            await asyncio.sleep(delay_seconds)
            
            # Check that iteration is still active
            current_iteration = db_manager.get_iteration(iteration.id)
            if not current_iteration:
                logger.info(f"Iteration {iteration.id} no longer exists, skipping scheduled iteration")
                db_manager.unlock_iteration(iteration.id)  # Cleanup lock
                return
                
            if current_iteration.status not in [IterationStatus.RUNNING, IterationStatus.WAITING_CI]:
                logger.info(f"Iteration {iteration.id} is no longer active (status: {current_iteration.status}), skipping scheduled iteration")
                db_manager.unlock_iteration(iteration.id)
                return
            
            # Check if we haven't reached max iterations
            if current_iteration.current_iteration >= current_iteration.max_iterations:
                logger.info(f"Max iterations reached for {iteration.id}, completing as failed")
                db_manager.unlock_iteration(iteration.id)
                await self._complete_iteration(current_iteration, IterationStatus.FAILED, "Max iterations reached")
                return
            
            # 🚨 CRITICAL: We should still have the lock from the continue method
            # Verify we still have the lock
            if not db_manager.is_iteration_locked(iteration.id):
                logger.error(f"🚨 CRITICAL: Lost lock on iteration {iteration.id} during scheduling!")
                logger.error(f"   This should not happen - lock should be maintained from continue method")
                return
            
            # Run the next iteration (lock is already held)
            logger.critical(f"🚀 Starting scheduled iteration for {current_iteration.repo_full_name}#{current_iteration.issue_number} (lock maintained)")
            await self._run_code_iteration(current_iteration)
            
        except Exception as e:
            logger.error(f"Failed to run scheduled iteration: {e}", exc_info=True)
            db_manager.unlock_iteration(iteration.id)
            await self._complete_iteration(iteration, IterationStatus.FAILED, f"Scheduled iteration failed: {str(e)}")

    async def _start_new_iteration_for_ci_failure(
        self,
        iteration: IssueIteration,
        ci_conclusion: str,
        ci_data_source: str
    ) -> None:
        """Legacy method - redirects to _continue_iteration_for_ci_failure."""
        logger.info("Redirecting to _continue_iteration_for_ci_failure for better PR continuity")
        await self._continue_iteration_for_ci_failure(iteration, ci_conclusion, ci_data_source)
    
    async def _analyze_ci_failure(self, iteration: IssueIteration, ci_conclusion: str, ci_data_source: str) -> Dict[str, Any]:
        """Analyze CI failure to understand what went wrong and suggest fixes."""
        try:
            logger.info(f"Analyzing CI failure for iteration {iteration.id}")
            
            analysis = {
                "ci_conclusion": ci_conclusion,
                "ci_data_source": ci_data_source,
                "potential_fixes": [],
                "failure_categories": [],
                "affected_files": []
            }
            
            # Get CI details if available
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    try:
                        # Get detailed CI status
                        ci_details = await github.get_pr_ci_status(owner, repo, iteration.pr_number)
                        
                        if ci_details and ci_details.get("failed_checks"):
                            failed_checks = ci_details["failed_checks"]
                            analysis["failed_checks"] = failed_checks
                            
                            # Categorize failures
                            for check in failed_checks:
                                check_lower = check.lower()
                                if "test" in check_lower:
                                    analysis["failure_categories"].append("test_failure")
                                    analysis["potential_fixes"].append("Fix failing tests")
                                elif "lint" in check_lower or "style" in check_lower:
                                    analysis["failure_categories"].append("linting_failure")
                                    analysis["potential_fixes"].append("Fix code style and linting issues")
                                elif "build" in check_lower or "compile" in check_lower:
                                    analysis["failure_categories"].append("build_failure")
                                    analysis["potential_fixes"].append("Fix compilation errors")
                                elif "security" in check_lower:
                                    analysis["failure_categories"].append("security_failure")
                                    analysis["potential_fixes"].append("Fix security vulnerabilities")
                        
                        # Get workflow run details if available
                        if ci_details.get("workflow_runs"):
                            for run in ci_details["workflow_runs"][:3]:  # Check last 3 runs
                                if run.get("conclusion") == "failure":
                                    # Try to get job details
                                    try:
                                        jobs = await github.get_workflow_run_jobs(owner, repo, run["id"])
                                        for job in jobs.get("jobs", []):
                                            if job.get("conclusion") == "failure":
                                                job_name = job.get("name", "").lower()
                                                if "test" in job_name:
                                                    analysis["failure_categories"].append("test_job_failure")
                                                elif "build" in job_name:
                                                    analysis["failure_categories"].append("build_job_failure")
                                    except Exception as job_e:
                                        logger.warning(f"Could not get job details: {job_e}")
                    
                    except Exception as ci_e:
                        logger.warning(f"Could not get detailed CI status: {ci_e}")
            
            # Add generic fixes based on conclusion
            if ci_conclusion == "failure":
                if not analysis["potential_fixes"]:
                    analysis["potential_fixes"].extend([
                        "Check and fix any compilation errors",
                        "Ensure all tests pass",
                        "Fix any linting or style issues",
                        "Verify all dependencies are correctly specified"
                    ])
            
            # Remove duplicates
            analysis["failure_categories"] = list(set(analysis["failure_categories"]))
            analysis["potential_fixes"] = list(set(analysis["potential_fixes"]))
            
            logger.info(f"CI failure analysis complete: {len(analysis['potential_fixes'])} potential fixes identified")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze CI failure: {e}", exc_info=True)
            return {
                "ci_conclusion": ci_conclusion,
                "ci_data_source": ci_data_source,
                "potential_fixes": ["Fix CI issues based on build logs"],
                "failure_categories": ["unknown"],
                "error": str(e)
            }
    
    async def _build_iteration_context(self, iteration: IssueIteration) -> Dict[str, Any]:
        """Build context from previous iterations to preserve learning."""
        try:
            logger.info(f"Building iteration context for iteration {iteration.id}")
            
            context = {
                "current_iteration": iteration.current_iteration,
                "max_iterations": iteration.max_iterations,
                "previous_changes": [],
                "previous_feedback": [],
                "files_modified_history": [],
                "ci_failure_history": []
            }
            
            # Get PR files to understand what was changed
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    try:
                        pr_files = await github.get_pull_request_files(owner, repo, iteration.pr_number)
                        
                        for file_info in pr_files:
                            context["files_modified_history"].append({
                                "filename": file_info["filename"],
                                "status": file_info["status"],
                                "additions": file_info["additions"],
                                "deletions": file_info["deletions"],
                                "changes": file_info["changes"]
                            })
                        
                        context["total_files_changed"] = len(pr_files)
                        context["total_additions"] = sum(f["additions"] for f in pr_files)
                        context["total_deletions"] = sum(f["deletions"] for f in pr_files)
                        
                    except Exception as pr_e:
                        logger.warning(f"Could not get PR files: {pr_e}")
            
            # Add previous review feedback if available
            if iteration.last_review_feedback:
                context["previous_feedback"].append({
                    "iteration": iteration.current_iteration,
                    "feedback": iteration.last_review_feedback,
                    "timestamp": iteration.updated_at
                })
            
            # Add CI failure history
            if iteration.last_ci_conclusion and iteration.last_ci_conclusion != "success":
                context["ci_failure_history"].append({
                    "iteration": iteration.current_iteration,
                    "conclusion": iteration.last_ci_conclusion,
                    "status": iteration.last_ci_status,
                    "timestamp": iteration.ci_status_updated_at
                })
            
            logger.info(f"Iteration context built: {context['total_files_changed']} files changed, {len(context['previous_feedback'])} feedback items")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build iteration context: {e}", exc_info=True)
            return {
                "current_iteration": iteration.current_iteration,
                "max_iterations": iteration.max_iterations,
                "error": str(e)
            }
    
    def _format_ci_failure_feedback(self, ci_analysis: Dict[str, Any], iteration_context: Dict[str, Any]) -> str:
        """Format enhanced CI failure feedback for the next iteration."""
        feedback_parts = []
        
        # CI failure summary
        feedback_parts.append(f"CI failed with status: {ci_analysis['ci_conclusion']} (source: {ci_analysis['ci_data_source']})")
        
        # Failure categories
        if ci_analysis.get("failure_categories"):
            categories = ", ".join(ci_analysis["failure_categories"])
            feedback_parts.append(f"Failure categories: {categories}")
        
        # Specific failed checks
        if ci_analysis.get("failed_checks"):
            failed_checks = ", ".join(ci_analysis["failed_checks"])
            feedback_parts.append(f"Failed checks: {failed_checks}")
        
        # Potential fixes
        if ci_analysis.get("potential_fixes"):
            fixes = "\n".join(f"- {fix}" for fix in ci_analysis["potential_fixes"])
            feedback_parts.append(f"Potential fixes:\n{fixes}")
        
        # Context from previous iterations
        if iteration_context.get("files_modified_history"):
            file_count = len(iteration_context["files_modified_history"])
            feedback_parts.append(f"Previous changes: {file_count} files modified in this PR")
            
            # List modified files
            modified_files = [f["filename"] for f in iteration_context["files_modified_history"]]
            if modified_files:
                files_list = ", ".join(modified_files[:5])  # Show first 5 files
                if len(modified_files) > 5:
                    files_list += f" (and {len(modified_files) - 5} more)"
                feedback_parts.append(f"Modified files: {files_list}")
        
        # Previous CI failures
        if iteration_context.get("ci_failure_history"):
            feedback_parts.append(f"This is iteration {iteration_context['current_iteration']} of {iteration_context['max_iterations']}")
            feedback_parts.append("Focus on incremental fixes rather than major rewrites")
        
        return ". ".join(feedback_parts) + "."
    
    def _format_ci_failure_comment(self, ci_analysis: Dict[str, Any], iteration_context: Dict[str, Any], next_iteration: int) -> str:
        """Format detailed CI failure comment for PR."""
        comment = f"""## 🔄 Starting Iteration {next_iteration} - CI Failure Analysis

### 🚨 CI Status: {ci_analysis['ci_conclusion'].upper()}
**Data Source:** {ci_analysis['ci_data_source']}

### 📊 Failure Analysis:"""
        
        if ci_analysis.get("failed_checks"):
            comment += f"""
**Failed Checks:** {', '.join(ci_analysis['failed_checks'])}"""
        
        if ci_analysis.get("failure_categories"):
            comment += f"""
**Categories:** {', '.join(ci_analysis['failure_categories'])}"""
        
        if ci_analysis.get("potential_fixes"):
            comment += f"""

### 🔧 Potential Fixes:
{chr(10).join(f'- {fix}' for fix in ci_analysis['potential_fixes'])}"""
        
        # Add context information
        if iteration_context.get("files_modified_history"):
            file_count = len(iteration_context["files_modified_history"])
            comment += f"""

### 📁 Current PR Context:
- **Files Modified:** {file_count}
- **Total Changes:** +{iteration_context.get('total_additions', 0)}/-{iteration_context.get('total_deletions', 0)}"""
        
        comment += f"""

### 🎯 Next Steps:
The agent will analyze the CI failure and make targeted fixes to resolve the issues. This is iteration {next_iteration} of {iteration_context.get('max_iterations', 'unknown')}.

**Strategy:** Focus on incremental fixes based on the failure analysis above."""
        
        return comment

    async def _complete_iteration(
        self,
        iteration: IssueIteration,
        status: IterationStatus,
        message: str
    ) -> None:
        """Complete iteration and close PR only when successfully completed - ANTI-DUPLICATION."""
        try:
            # 🚨 CRITICAL: Check if iteration is already completed to prevent duplication
            current_iteration = db_manager.get_iteration(iteration.id)
            if not current_iteration or not current_iteration.is_active:
                logger.warning(f"🚫 COMPLETION DUPLICATION BLOCKED: Iteration {iteration.id} already completed")
                logger.warning(f"   📋 Current status: {current_iteration.status if current_iteration else 'not found'}")
                logger.warning(f"   🔄 This prevents duplicate completion comments")
                return
            
            logger.critical(f"🎯 COMPLETING ITERATION {iteration.id}")
            logger.critical(f"   📊 Status: {status}")
            logger.critical(f"   💬 Message: {message}")
            
            # Complete iteration in database
            db_manager.complete_iteration(iteration.id, status)
            
            # Always unlock iteration when completing
            db_manager.unlock_iteration(iteration.id)
            logger.critical(f"🔓 ITERATION {iteration.id} COMPLETED AND UNLOCKED")
            
            # Post final comment to PR if exists
            if iteration.pr_number:
                async with await get_github_client(iteration.installation_id) as github:
                    owner, repo = iteration.repo_full_name.split("/")
                    
                    if status == IterationStatus.COMPLETED:
                        # 🚨 CRITICAL: PR CLOSURE MONITORING - Log any potential closure attempts
                        logger.critical(f"🚨 PR CLOSURE CHECKPOINT: Would complete PR #{iteration.pr_number} as SUCCESS")
                        logger.critical(f"   📊 Status: {status}")
                        logger.critical(f"   💬 Message: {message}")
                        logger.critical(f"   🔒 PR Closure: DISABLED (manual intervention required)")
                        
                        final_comment = f"""## ✅ SDLC Cycle Completed Successfully!

The automated development cycle has been completed successfully after {iteration.current_iteration} iteration(s).

**Final Status:** {message}

**🚨 PR STATUS:** This PR remains OPEN for manual review and merge.
**🔧 SAFETY MODE:** Automatic PR closure is DISABLED to prevent incorrect closures.
**📋 Action Required:** Please manually review and merge this PR if appropriate.

**Completion Details:**
- Iteration: {iteration.current_iteration}/{iteration.max_iterations}
- Status: {status}
- Message: {message}
- Timestamp: {datetime.utcnow().isoformat()}
- PR Head SHA: {iteration.pr_head_sha[:8] if iteration.pr_head_sha else 'unknown'}"""
                        
                        # Post comment but NEVER close PR
                        await github.create_issue_comment(
                            owner, repo, iteration.pr_number, final_comment
                        )
                        
                        logger.critical(f"🚨 WOULD HAVE CLOSED PR #{iteration.pr_number} but closure is DISABLED (debug mode)")
                        logger.info(f"COMPLETED iteration {iteration.id} - PR #{iteration.pr_number} LEFT OPEN for manual intervention")
                            
                    else:  # FAILED
                        # When failed - DON'T close PR, leave open for manual intervention
                        final_comment = f"""## ❌ SDLC Cycle Failed - Manual Intervention Required

The automated development cycle could not be completed successfully.

**Reason:** {message}
**Iterations:** {iteration.current_iteration}/{iteration.max_iterations}

**PR Status:** This PR remains OPEN for manual review and fixes.
**IMPORTANT:** The agent will NOT make further automatic changes to this PR.
Manual intervention is required to resolve the remaining issues."""
                        
                        # Post comment but DON'T close PR - let humans fix it
                        await github.create_issue_comment(
                            owner, repo, iteration.pr_number, final_comment
                        )
                        
                        logger.info(f"Posted failure comment to PR #{iteration.pr_number} but LEFT IT OPEN for manual intervention")
            
            logger.info(f"Completed iteration {iteration.id} with status {status}: {message}")
            if status == IterationStatus.FAILED:
                logger.info(f"PR #{iteration.pr_number} left OPEN for manual intervention")
            elif status == IterationStatus.COMPLETED:
                logger.info(f"🚨 DEBUG: PR #{iteration.pr_number} completion processed - PR LEFT OPEN (closure disabled)")
            
        except Exception as e:
            logger.error(f"Failed to complete iteration: {e}", exc_info=True)
            # Always unlock on error
            db_manager.unlock_iteration(iteration.id)


    async def _check_for_unhandled_ci_events(self, github, owner: str, repo: str, pr_number: int) -> bool:
        """Проверить наличие необработанных CI событий."""
        try:
            # Получить recent workflow runs для PR
            workflow_runs = await github.get_workflow_runs(owner, repo, pr_number)
            
            # Проверить, есть ли runs без соответствующих записей в БД
            # (это упрощённая проверка - в реальности нужно сравнивать с БД)
            
            recent_runs = workflow_runs.get("workflow_runs", [])[:5]  # Последние 5 runs
            
            for run in recent_runs:
                if run.get("status") == "completed" and run.get("conclusion") in ["failure", "success"]:
                    # Здесь можно добавить проверку, был ли этот run обработан
                    # Пока просто логируем
                    logger.debug(f"Found completed workflow run: {run.get('id')} with conclusion: {run.get('conclusion')}")
            
            # Возвращаем False пока не реализована полная проверка
            return False
            
        except Exception as e:
            logger.warning(f"Could not check for unhandled CI events: {e}")
            return False

    async def _complete_iteration_with_file_not_found_error(self, context: Dict[str, Any], file_path: str, analysis: Dict) -> None:
        """Завершить итерацию с ошибкой 'файл не найден'."""
        try:
            # Собрать информацию о поиске
            search_results = analysis.get("existence_check_results", [])
            
            error_message = f"""File not found: {file_path}

Task type 'add_comments' requires existing files to modify.

Search results:
"""
            
            for result in search_results:
                error_message += f"- {result.get('intended_file', 'unknown')}: {result.get('action', 'unknown')} - {result.get('reason', 'no reason')}\n"
            
            if not search_results:
                error_message += "- No search results available\n"
            
            error_message += """
Please verify the file path or entity name in the issue description.
Available files in repository can be found in the repository analysis above.
"""
            
            # Логировать ошибку
            logger.error(f"File not found error for add_comments task: {file_path}")
            
            # Если есть PR, добавить комментарий
            if context.get("pr_number"):
                try:
                    async with await get_github_client(context.get("installation_id")) as github:
                        owner, repo = context["repo_full_name"].split("/")
                        await github.create_issue_comment(
                            owner, repo, context["pr_number"],
                            f"## ❌ File Not Found Error\n\n{error_message}"
                        )
                except Exception as comment_e:
                    logger.warning(f"Could not post file not found comment: {comment_e}")
            
        except Exception as e:
            logger.error(f"Failed to complete iteration with file not found error: {e}")


# Global orchestrator instance
orchestrator = SDLCOrchestrator()
