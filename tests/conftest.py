"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.repository_analyzer import RepositoryMap, ClassInfo, ModuleInfo, FunctionInfo


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client for testing."""
    client = AsyncMock()
    
    # Default mock responses
    client.list_repository_files.return_value = [
        "app/main.py",
        "app/models.py",
        "app/services.py",
        "app/utils.py",
        "tests/test_main.py"
    ]
    
    client.get_file_content.return_value = {
        "content": "sample_content".encode().hex(),
        "sha": "abc123"
    }
    
    client.get_issue.return_value = {
        "title": "Test Issue",
        "body": "Test issue description",
        "number": 123
    }
    
    client.get_pull_request.return_value = {
        "title": "Test PR",
        "body": "Test PR description",
        "number": 456,
        "head": {"sha": "def456"}
    }
    
    client.get_pull_request_files.return_value = [
        {
            "filename": "app/services.py",
            "status": "modified",
            "additions": 10,
            "deletions": 2,
            "changes": 12,
            "patch": "diff content"
        }
    ]
    
    return client


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = AsyncMock()
    
    client.create_system_message.return_value = {
        "role": "system",
        "content": "You are a helpful assistant."
    }
    
    client.create_user_message.return_value = {
        "role": "user", 
        "content": "Test user message"
    }
    
    client.generate_response.return_value = '''
    {
        "summary": "Test implementation",
        "files_to_modify": ["app/services.py"],
        "files_to_create": [],
        "requirements": ["Add test functionality"],
        "technical_approach": "Enhance existing service",
        "dependencies": [],
        "architecture_notes": "Follow existing patterns",
        "existing_classes_to_enhance": ["TestService"]
    }
    '''
    
    return client


@pytest.fixture
def sample_repository_map():
    """Create a sample repository map for testing."""
    user_service = ClassInfo(
        name="UserService",
        file_path="app/services.py",
        methods=["create_user", "get_user", "update_user", "delete_user"],
        dependencies=["db_client", "logger"],
        purpose="Service for managing user operations"
    )
    
    auth_service = ClassInfo(
        name="AuthService",
        file_path="app/auth.py",
        methods=["login", "logout", "validate_token"],
        dependencies=["jwt_handler", "user_service"],
        purpose="Service for handling authentication"
    )
    
    helper_function = FunctionInfo(
        name="validate_email",
        file_path="app/utils.py",
        parameters=["email"],
        return_type="bool",
        purpose="Validate email format"
    )
    
    services_module = ModuleInfo(
        file_path="app/services.py",
        classes=[user_service],
        functions=[],
        imports=["asyncio", "typing", "logging"],
        dependencies=["app.database", "app.models"]
    )
    
    auth_module = ModuleInfo(
        file_path="app/auth.py",
        classes=[auth_service],
        functions=[],
        imports=["jwt", "datetime"],
        dependencies=["app.services"]
    )
    
    utils_module = ModuleInfo(
        file_path="app/utils.py",
        classes=[],
        functions=[helper_function],
        imports=["re", "typing"],
        dependencies=[]
    )
    
    return RepositoryMap(
        modules={
            "app/services.py": services_module,
            "app/auth.py": auth_module,
            "app/utils.py": utils_module
        },
        classes={
            "UserService": user_service,
            "AuthService": auth_service
        },
        functions={
            "validate_email": helper_function
        },
        dependencies={
            "app/services.py": ["app.database", "app.models"],
            "app/auth.py": ["app.services"],
            "app/utils.py": []
        },
        architecture_patterns=["Service Layer Pattern", "MVC Pattern"]
    )


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing parsing."""
    return '''
"""Sample service module."""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserModel:
    """User data model."""
    id: int
    name: str
    email: str
    
    def validate(self) -> bool:
        """Validate user data."""
        return bool(self.name and self.email and '@' in self.email)


class UserService:
    """Service for managing users."""
    
    def __init__(self, db_client, logger=None):
        """Initialize user service."""
        self.db_client = db_client
        self.logger = logger or logging.getLogger(__name__)
    
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[UserModel]:
        """Create a new user."""
        try:
            user = UserModel(**user_data)
            if user.validate():
                await self.db_client.save(user)
                self.logger.info(f"Created user: {user.email}")
                return user
            else:
                self.logger.warning("Invalid user data provided")
                return None
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return None
    
    async def get_user(self, user_id: int) -> Optional[UserModel]:
        """Get user by ID."""
        try:
            return await self.db_client.get(UserModel, user_id)
        except Exception as e:
            self.logger.error(f"Error getting user {user_id}: {e}")
            return None
    
    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        try:
            user = await self.get_user(user_id)
            if user:
                for key, value in updates.items():
                    setattr(user, key, value)
                if user.validate():
                    await self.db_client.save(user)
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating user {user_id}: {e}")
            return False
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user by ID."""
        try:
            return await self.db_client.delete(UserModel, user_id)
        except Exception as e:
            self.logger.error(f"Error deleting user {user_id}: {e}")
            return False


def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_user_name(first_name: str, last_name: str) -> str:
    """Format user's full name."""
    return f"{first_name.strip()} {last_name.strip()}".strip()
'''


@pytest.fixture
def mock_iteration():
    """Create a mock iteration for testing."""
    iteration = MagicMock()
    iteration.id = 1
    iteration.repo_full_name = "owner/repo"
    iteration.issue_number = 123
    iteration.issue_title = "Test Issue"
    iteration.issue_body = "Test issue description"
    iteration.current_iteration = 1
    iteration.max_iterations = 3
    iteration.installation_id = 456
    iteration.branch_name = "agent/issue-123"
    iteration.pr_number = None
    iteration.last_review_feedback = None
    iteration.status = "RUNNING"
    iteration.last_ci_status = None
    iteration.last_ci_conclusion = None
    iteration.pr_head_sha = None
    iteration.ci_status_source = None
    iteration.ci_status_updated_at = None
    return iteration


@pytest.fixture
def sample_analysis():
    """Create a sample analysis result for testing."""
    return {
        "summary": "Add user authentication functionality",
        "files_to_modify": ["app/services.py"],
        "files_to_create": [],
        "requirements": [
            "Add login method to UserService",
            "Add password validation",
            "Add session management"
        ],
        "technical_approach": "Enhance existing UserService class with authentication methods",
        "dependencies": ["bcrypt", "jwt"],
        "architecture_notes": "Follow existing service layer pattern and maintain consistency",
        "existing_classes_to_enhance": ["UserService"]
    }


@pytest.fixture
def sample_review_context():
    """Create a sample review context for testing."""
    return {
        "repo_full_name": "owner/repo",
        "issue_number": 123,
        "issue_title": "Add authentication",
        "issue_body": "Add user authentication functionality",
        "pr_number": 456,
        "pr_data": {
            "title": "Add user authentication",
            "body": "Implements login and logout functionality",
            "head": {"sha": "abc123"}
        },
        "pr_files": [
            {
                "filename": "app/services.py",
                "status": "modified",
                "additions": 25,
                "deletions": 3,
                "changes": 28,
                "patch": "diff content here"
            }
        ],
        "ci_status": "completed",
        "ci_conclusion": "success",
        "ci_data_source": "webhook",
        "iteration": 1,
        "installation_id": 456
    }