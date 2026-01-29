"""Tests for orchestrator architecture-aware functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.orchestrator import SDLCOrchestrator
from app.database import IssueIteration, IterationStatus
from app.repository_analyzer import RepositoryMap, ClassInfo, ModuleInfo


class TestSDLCOrchestratorArchitecture:
    """Test cases for architecture-aware orchestrator functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create an SDLCOrchestrator instance."""
        with patch('app.orchestrator.settings') as mock_settings:
            mock_settings.use_yandex_gpt = False
            mock_settings.openai_api_key = "test_key"
            mock_settings.openai_model = "gpt-4"
            mock_settings.yandex_gpt_api_key = None
            mock_settings.yandex_gpt_folder_id = None
            return SDLCOrchestrator()
    
    @pytest.fixture
    def mock_iteration(self):
        """Create a mock iteration."""
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.repo_full_name = "owner/repo"
        iteration.issue_number = 123
        iteration.issue_title = "Add user authentication"
        iteration.issue_body = "Need to add login functionality"
        iteration.current_iteration = 1
        iteration.max_iterations = 3
        iteration.installation_id = 456
        iteration.branch_name = "agent/issue-123"
        iteration.pr_number = None
        iteration.last_review_feedback = None
        return iteration
    
    @pytest.fixture
    def sample_repository_map(self):
        """Create a sample repository map."""
        return RepositoryMap(
            modules={
                "app/services.py": ModuleInfo(
                    file_path="app/services.py",
                    classes=[
                        ClassInfo(
                            name="UserService",
                            file_path="app/services.py",
                            methods=["create_user", "get_user"],
                            dependencies=["db_client"],
                            purpose="Service for managing users"
                        )
                    ],
                    functions=[],
                    imports=["asyncio", "typing"],
                    dependencies=[]
                )
            },
            classes={
                "UserService": ClassInfo(
                    name="UserService",
                    file_path="app/services.py",
                    methods=["create_user", "get_user"],
                    dependencies=["db_client"],
                    purpose="Service for managing users"
                )
            },
            functions={},
            dependencies={},
            architecture_patterns=["Service Layer Pattern"]
        )
    
    @pytest.mark.asyncio
    async def test_analyze_issue_requirements_with_architecture(self, orchestrator, sample_repository_map):
        """Test issue analysis with repository architecture context."""
        context = {
            "repo_full_name": "owner/repo",
            "issue_number": 123,
            "issue_title": "Add user authentication",
            "issue_body": "Need to add login functionality to existing user service",
            "iteration": 1,
            "installation_id": 456
        }
        
        # Mock repository analyzer
        with patch('app.orchestrator.RepositoryAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_repository.return_value = sample_repository_map
            mock_analyzer.find_relevant_classes.return_value = [sample_repository_map.classes["UserService"]]
            mock_analyzer.suggest_modification_targets.return_value = []
            
            # Mock GitHub client
            with patch('app.orchestrator.get_github_client') as mock_get_client:
                mock_github = AsyncMock()
                mock_get_client.return_value.__aenter__.return_value = mock_github
                
                # Mock LLM response
                orchestrator.llm_client.generate_response = AsyncMock(return_value='''
                {
                    "summary": "Add authentication to existing UserService",
                    "files_to_modify": ["app/services.py"],
                    "files_to_create": [],
                    "requirements": ["Add login method", "Add password validation"],
                    "technical_approach": "Enhance existing UserService class with authentication methods",
                    "dependencies": ["bcrypt"],
                    "architecture_notes": "Follow existing service layer pattern",
                    "existing_classes_to_enhance": ["UserService"]
                }
                ''')
                
                result = await orchestrator._analyze_issue_requirements(context)
        
        assert result is not None
        assert result["files_to_modify"] == ["app/services.py"]
        assert result["files_to_create"] == []
        assert "UserService" in result["existing_classes_to_enhance"]
        assert result["repository_map"] == sample_repository_map
        assert len(result["relevant_classes"]) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_issue_requirements_fallback_to_basic(self, orchestrator):
        """Test fallback to basic analysis when repository analysis fails."""
        context = {
            "repo_full_name": "owner/repo",
            "issue_number": 123,
            "issue_title": "Add new feature",
            "issue_body": "Add some new functionality",
            "iteration": 1,
            "installation_id": 456
        }
        
        # Mock repository analyzer to fail
        with patch('app.orchestrator.RepositoryAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_repository.return_value = None  # Analysis fails
            
            # Mock GitHub client
            with patch('app.orchestrator.get_github_client') as mock_get_client:
                mock_github = AsyncMock()
                mock_get_client.return_value.__aenter__.return_value = mock_github
                
                # Mock basic analysis
                with patch.object(orchestrator, '_analyze_issue_requirements_basic') as mock_basic:
                    mock_basic.return_value = {
                        "summary": "Basic analysis result",
                        "files_to_create": ["new_file.py"]
                    }
                    
                    result = await orchestrator._analyze_issue_requirements(context)
        
        assert result is not None
        assert result["summary"] == "Basic analysis result"
        mock_basic.assert_called_once_with(context)
    
    @pytest.mark.asyncio
    async def test_generate_file_content_with_architecture_awareness(self, orchestrator, sample_repository_map):
        """Test file content generation with architecture awareness."""
        analysis = {
            "summary": "Add authentication methods",
            "technical_approach": "Enhance existing UserService",
            "requirements": ["Add login method"],
            "architecture_notes": "Follow service layer pattern",
            "existing_classes_to_enhance": ["UserService"],
            "repository_map": sample_repository_map,
            "relevant_classes": [sample_repository_map.classes["UserService"]],
            "modification_targets": []
        }
        
        context = {
            "issue_number": 123,
            "iteration": 1
        }
        
        current_content = '''
class UserService:
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        pass
'''
        
        # Mock CodeModifier
        with patch('app.orchestrator.CodeModifier') as mock_modifier_class:
            mock_modifier = AsyncMock()
            mock_modifier_class.return_value = mock_modifier
            mock_modifier.modify_existing_file.return_value = '''
class UserService:
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        pass
    
    async def authenticate_user(self, email, password):
        # New authentication method
        return True
'''
            
            result = await orchestrator._generate_file_content(
                file_path="app/services.py",
                current_content=current_content,
                analysis=analysis,
                context=context,
                is_modification=True
            )
        
        assert result is not None
        assert "authenticate_user" in result
        mock_modifier.modify_existing_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_file_content_fallback_to_basic(self, orchestrator, sample_repository_map):
        """Test fallback to basic generation when architecture-aware modification fails."""
        analysis = {
            "summary": "Add authentication methods",
            "repository_map": sample_repository_map,
            "relevant_classes": [sample_repository_map.classes["UserService"]],
            "modification_targets": []
        }
        
        context = {"issue_number": 123, "iteration": 1}
        current_content = "class UserService: pass"
        
        # Mock CodeModifier to fail
        with patch('app.orchestrator.CodeModifier') as mock_modifier_class:
            mock_modifier = AsyncMock()
            mock_modifier_class.return_value = mock_modifier
            mock_modifier.modify_existing_file.return_value = None  # Modification fails
            
            # Mock LLM for fallback
            orchestrator.llm_client.generate_response = AsyncMock(return_value='''
class UserService:
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def authenticate_user(self, email, password):
        return True
''')
            
            result = await orchestrator._generate_file_content(
                file_path="app/services.py",
                current_content=current_content,
                analysis=analysis,
                context=context,
                is_modification=True
            )
        
        assert result is not None
        assert "authenticate_user" in result
        # Should have tried architecture-aware modification first
        mock_modifier.modify_existing_file.assert_called_once()
        # Then fallen back to LLM
        orchestrator.llm_client.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_reviewer_agent_with_architecture(self, orchestrator, sample_repository_map):
        """Test reviewer agent execution with architecture awareness."""
        context = {
            "repo_full_name": "owner/repo",
            "installation_id": 456,
            "issue_title": "Add authentication",
            "issue_body": "Add login functionality",
            "issue_number": 123,
            "pr_data": {"title": "Test PR"},
            "pr_files": [
                {
                    "filename": "app/services.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 2,
                    "changes": 12,
                    "patch": "diff content"
                }
            ],
            "ci_conclusion": "success"
        }
        
        # Mock repository analyzer
        with patch('app.orchestrator.RepositoryAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_repository.return_value = sample_repository_map
            
            # Mock GitHub client and auth
            with patch('app.orchestrator.get_github_client') as mock_get_client, \
                 patch('app.orchestrator.github_app_auth') as mock_auth, \
                 patch('ai_coding_agent.github_client.GitHubClient') as mock_old_client, \
                 patch('ai_coding_agent.reviewer_agent.ReviewerAgent') as mock_reviewer_class:
                
                mock_github = AsyncMock()
                mock_get_client.return_value.__aenter__.return_value = mock_github
                mock_auth.get_installation_token.return_value = "test_token"
                
                # Mock reviewer agent
                mock_reviewer = AsyncMock()
                mock_reviewer_class.return_value = mock_reviewer
                mock_reviewer._perform_comprehensive_review.return_value = {
                    "overall_assessment": {
                        "score": 85,
                        "recommendation": "approve_with_suggestions",
                        "summary": "Good implementation"
                    },
                    "code_quality": {"summary": "Good"},
                    "requirements_compliance": {"summary": "Compliant"},
                    "security_analysis": {"summary": "Secure"}
                }
                
                result = await orchestrator._execute_reviewer_agent(context)
        
        assert result is not None
        assert "architecture_analysis" in result
        assert "repository_context" in result["architecture_analysis"]
        assert result["architecture_analysis"]["repository_context"]["total_modules"] == 1
        assert "Service Layer Pattern" in result["architecture_analysis"]["repository_context"]["architecture_patterns"]
    
    @pytest.mark.asyncio
    async def test_execute_reviewer_agent_without_architecture(self, orchestrator):
        """Test reviewer agent execution when repository analysis fails."""
        context = {
            "repo_full_name": "owner/repo",
            "installation_id": 456,
            "issue_title": "Add feature",
            "issue_body": "Add some feature",
            "issue_number": 123,
            "pr_data": {"title": "Test PR"},
            "pr_files": [],
            "ci_conclusion": "success"
        }
        
        # Mock repository analyzer to fail
        with patch('app.orchestrator.RepositoryAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_repository.return_value = None  # Analysis fails
            
            # Mock GitHub client and auth
            with patch('app.orchestrator.get_github_client') as mock_get_client, \
                 patch('app.orchestrator.github_app_auth') as mock_auth, \
                 patch('ai_coding_agent.github_client.GitHubClient') as mock_old_client, \
                 patch('ai_coding_agent.reviewer_agent.ReviewerAgent') as mock_reviewer_class:
                
                mock_github = AsyncMock()
                mock_get_client.return_value.__aenter__.return_value = mock_github
                mock_auth.get_installation_token.return_value = "test_token"
                
                # Mock reviewer agent
                mock_reviewer = AsyncMock()
                mock_reviewer_class.return_value = mock_reviewer
                mock_reviewer._perform_comprehensive_review.return_value = {
                    "overall_assessment": {
                        "score": 80,
                        "recommendation": "approve",
                        "summary": "Good implementation"
                    }
                }
                
                result = await orchestrator._execute_reviewer_agent(context)
        
        assert result is not None
        # Should not have architecture analysis when repository analysis fails
        assert "architecture_analysis" not in result or not result.get("architecture_analysis", {}).get("repository_context")
    
    def test_format_relevant_classes(self, orchestrator):
        """Test formatting relevant classes for LLM prompt."""
        relevant_classes = [
            ClassInfo(
                name="UserService",
                file_path="app/services.py",
                methods=["create_user", "get_user", "update_user", "delete_user", "authenticate_user", "validate_user"],
                dependencies=["db_client", "logger"],
                purpose="Service for managing user operations and authentication"
            ),
            ClassInfo(
                name="AuthService",
                file_path="app/auth.py",
                methods=["login", "logout"],
                dependencies=["jwt_handler"],
                purpose="Service for handling authentication"
            )
        ]
        
        result = orchestrator._format_relevant_classes(relevant_classes)
        
        assert "UserService" in result
        assert "AuthService" in result
        assert "app/services.py" in result
        assert "Service for managing user operations" in result
        assert "create_user, get_user, update_user, delete_user, authenticate_user" in result
        assert "(and 1 more)" in result  # Should truncate methods list
        assert "db_client, logger" in result  # Should show dependencies
    
    def test_format_relevant_classes_empty(self, orchestrator):
        """Test formatting empty relevant classes list."""
        result = orchestrator._format_relevant_classes([])
        
        assert result == "No directly relevant classes found."
    
    def test_format_modification_targets(self, orchestrator):
        """Test formatting modification targets for LLM prompt."""
        from app.repository_analyzer import ModificationTarget
        
        targets = [
            ModificationTarget(
                target_type="class",
                name="UserService",
                file_path="app/services.py",
                reason="Add authentication methods to existing user management service",
                confidence=0.9
            ),
            ModificationTarget(
                target_type="function",
                name="validate_user",
                file_path="app/utils.py",
                reason="Enhance validation logic",
                confidence=0.7
            )
        ]
        
        result = orchestrator._format_modification_targets(targets)
        
        assert "class: UserService" in result
        assert "function: validate_user" in result
        assert "app/services.py" in result
        assert "Add authentication methods" in result
        assert "0.9" in result
        assert "0.7" in result
    
    def test_format_modification_targets_empty(self, orchestrator):
        """Test formatting empty modification targets list."""
        result = orchestrator._format_modification_targets([])
        
        assert result == "No specific modification targets identified."


@pytest.mark.integration
class TestOrchestratorArchitectureIntegration:
    """Integration tests for orchestrator architecture functionality."""
    
    @pytest.mark.asyncio
    async def test_full_architecture_aware_workflow(self):
        """Test the complete architecture-aware workflow."""
        # This would be a comprehensive integration test
        # that tests the full flow from issue analysis to code generation
        # with architecture awareness
        pass  # Placeholder for full integration test