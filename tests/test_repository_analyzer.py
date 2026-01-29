"""Tests for repository analyzer functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.repository_analyzer import RepositoryAnalyzer, RepositoryMap, ClassInfo, FunctionInfo, ModuleInfo


class TestRepositoryAnalyzer:
    """Test cases for RepositoryAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a RepositoryAnalyzer instance."""
        return RepositoryAnalyzer()
    
    @pytest.fixture
    def mock_github_client(self):
        """Create a mock GitHub client."""
        client = AsyncMock()
        client.list_repository_files.return_value = [
            "app/main.py",
            "app/models.py", 
            "app/utils.py",
            "tests/test_main.py",
            "README.md"
        ]
        return client
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''
"""Sample module for testing."""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class UserModel:
    """User data model."""
    id: int
    name: str
    email: str
    
    def validate(self) -> bool:
        """Validate user data."""
        return bool(self.name and self.email)


class UserService:
    """Service for managing users."""
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[UserModel]:
        """Create a new user."""
        try:
            user = UserModel(**user_data)
            if user.validate():
                # Save to database
                await self.db_client.save(user)
                return user
            return None
        except Exception as e:
            print(f"Error creating user: {e}")
            return None
    
    async def get_user(self, user_id: int) -> Optional[UserModel]:
        """Get user by ID."""
        return await self.db_client.get(UserModel, user_id)


def helper_function(data: str) -> str:
    """Helper function for data processing."""
    return data.strip().lower()
'''
    
    @pytest.mark.asyncio
    async def test_analyze_repository_success(self, analyzer, mock_github_client, sample_python_code):
        """Test successful repository analysis."""
        # Mock file content retrieval
        mock_github_client.get_file_content.return_value = {
            "content": sample_python_code.encode().hex()  # Simulate base64 encoding
        }
        
        with patch('base64.b64decode', return_value=sample_python_code.encode()):
            result = await analyzer.analyze_repository(mock_github_client, "owner", "repo")
        
        assert result is not None
        assert isinstance(result, RepositoryMap)
        assert len(result.modules) > 0
        
        # Check that Python files were processed
        python_files = [f for f in result.modules.keys() if f.endswith('.py')]
        assert len(python_files) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_repository_with_no_python_files(self, analyzer, mock_github_client):
        """Test repository analysis with no Python files."""
        mock_github_client.list_repository_files.return_value = [
            "README.md",
            "package.json",
            "Dockerfile"
        ]
        
        result = await analyzer.analyze_repository(mock_github_client, "owner", "repo")
        
        assert result is not None
        assert len(result.modules) == 0
        assert result.architecture_patterns == []
    
    def test_parse_python_file_success(self, analyzer, sample_python_code):
        """Test successful Python file parsing."""
        result = analyzer._parse_python_file(sample_python_code, "app/models.py")
        
        assert result is not None
        assert isinstance(result, ModuleInfo)
        assert result.file_path == "app/models.py"
        assert len(result.classes) >= 2  # UserModel and UserService
        assert len(result.functions) >= 1  # helper_function
        
        # Check class parsing
        user_service = next((cls for cls in result.classes if cls.name == "UserService"), None)
        assert user_service is not None
        assert "create_user" in user_service.methods
        assert "get_user" in user_service.methods
        assert user_service.purpose  # Should have extracted purpose from docstring
    
    def test_parse_python_file_with_syntax_error(self, analyzer):
        """Test parsing Python file with syntax error."""
        invalid_code = "def invalid_function(\n    # Missing closing parenthesis"
        
        result = analyzer._parse_python_file(invalid_code, "invalid.py")
        
        assert result is None
    
    def test_detect_architecture_patterns(self, analyzer):
        """Test architecture pattern detection."""
        modules = {
            "app/models.py": ModuleInfo(
                file_path="app/models.py",
                classes=[ClassInfo(name="UserModel", file_path="app/models.py", methods=[], dependencies=[], purpose="Data model")],
                functions=[],
                imports=["dataclasses"],
                dependencies=[]
            ),
            "app/services.py": ModuleInfo(
                file_path="app/services.py", 
                classes=[ClassInfo(name="UserService", file_path="app/services.py", methods=[], dependencies=[], purpose="Service layer")],
                functions=[],
                imports=[],
                dependencies=[]
            ),
            "app/controllers.py": ModuleInfo(
                file_path="app/controllers.py",
                classes=[ClassInfo(name="UserController", file_path="app/controllers.py", methods=[], dependencies=[], purpose="Controller")],
                functions=[],
                imports=[],
                dependencies=[]
            )
        }
        
        patterns = analyzer._detect_architecture_patterns(modules)
        
        assert "MVC Pattern" in patterns or "Service Layer Pattern" in patterns
    
    def test_find_relevant_classes(self, analyzer):
        """Test finding relevant classes based on description."""
        repository_map = RepositoryMap(
            modules={},
            classes={
                "UserService": ClassInfo(
                    name="UserService",
                    file_path="app/services.py",
                    methods=["create_user", "get_user"],
                    dependencies=[],
                    purpose="Service for managing users"
                ),
                "OrderService": ClassInfo(
                    name="OrderService", 
                    file_path="app/services.py",
                    methods=["create_order"],
                    dependencies=[],
                    purpose="Service for managing orders"
                )
            },
            functions={},
            dependencies={},
            architecture_patterns=[]
        )
        
        # Test finding user-related classes
        relevant = analyzer.find_relevant_classes(repository_map, "Add user authentication feature")
        
        assert len(relevant) > 0
        user_service = next((cls for cls in relevant if cls.name == "UserService"), None)
        assert user_service is not None
    
    def test_suggest_modification_targets(self, analyzer):
        """Test suggesting modification targets."""
        repository_map = RepositoryMap(
            modules={},
            classes={
                "UserService": ClassInfo(
                    name="UserService",
                    file_path="app/services.py", 
                    methods=["create_user", "get_user"],
                    dependencies=[],
                    purpose="Service for managing users"
                )
            },
            functions={},
            dependencies={},
            architecture_patterns=[]
        )
        
        targets = analyzer.suggest_modification_targets(repository_map, "Add user password reset functionality")
        
        assert len(targets) > 0
        user_target = next((t for t in targets if "UserService" in t.name), None)
        assert user_target is not None
        assert user_target.confidence > 0.5


class TestRepositoryMap:
    """Test cases for RepositoryMap."""
    
    def test_get_structure_summary(self):
        """Test getting repository structure summary."""
        repository_map = RepositoryMap(
            modules={
                "app/main.py": ModuleInfo("app/main.py", [], [], [], [])
            },
            classes={
                "UserService": ClassInfo("UserService", "app/services.py", [], [], "User management")
            },
            functions={},
            dependencies={},
            architecture_patterns=["MVC Pattern"]
        )
        
        summary = repository_map.get_structure_summary()
        
        assert "1 modules" in summary
        assert "1 classes" in summary
        assert "MVC Pattern" in summary
    
    def test_find_relevant_classes_empty(self):
        """Test finding relevant classes with empty repository."""
        repository_map = RepositoryMap({}, {}, {}, {}, [])
        analyzer = RepositoryAnalyzer()
        
        relevant = analyzer.find_relevant_classes(repository_map, "test description")
        
        assert len(relevant) == 0


@pytest.mark.integration
class TestRepositoryAnalyzerIntegration:
    """Integration tests for RepositoryAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test the complete analysis workflow."""
        analyzer = RepositoryAnalyzer()
        mock_github = AsyncMock()
        
        # Mock repository structure
        mock_github.list_repository_files.return_value = [
            "app/main.py",
            "app/models.py",
            "app/services.py"
        ]
        
        # Mock file contents
        sample_code = '''
class TestService:
    """Test service class."""
    
    def __init__(self):
        pass
    
    async def process_data(self, data):
        """Process some data."""
        return data
'''
        
        mock_github.get_file_content.return_value = {
            "content": sample_code.encode().hex()
        }
        
        with patch('base64.b64decode', return_value=sample_code.encode()):
            result = await analyzer.analyze_repository(mock_github, "test", "repo")
        
        assert result is not None
        assert len(result.modules) == 3  # All Python files processed
        assert len(result.classes) > 0