"""Tests for code modifier functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.code_modifier import CodeModifier
from app.repository_analyzer import RepositoryMap, ClassInfo, ModuleInfo


class TestCodeModifier:
    """Test cases for CodeModifier."""
    
    @pytest.fixture
    def modifier(self):
        """Create a CodeModifier instance."""
        return CodeModifier()
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.create_system_message.return_value = {"role": "system", "content": "test"}
        client.create_user_message.return_value = {"role": "user", "content": "test"}
        client.generate_response.return_value = '''
class UserService:
    """Enhanced user service with new functionality."""
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        """Create a new user."""
        # Existing functionality preserved
        pass
    
    async def authenticate_user(self, email, password):
        """New authentication method."""
        # New functionality added
        return True
'''
        return client
    
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
    
    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis result."""
        return {
            "summary": "Add user authentication functionality",
            "files_to_modify": ["app/services.py"],
            "files_to_create": [],
            "requirements": ["Add login method", "Add password validation"],
            "technical_approach": "Enhance existing UserService class",
            "dependencies": ["bcrypt"],
            "architecture_notes": "Follow existing service pattern",
            "existing_classes_to_enhance": ["UserService"]
        }
    
    @pytest.fixture
    def sample_current_content(self):
        """Sample current file content."""
        return '''
"""User service module."""

import asyncio
from typing import Optional


class UserService:
    """Service for managing users."""
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        """Create a new user."""
        # Implementation here
        pass
    
    async def get_user(self, user_id):
        """Get user by ID."""
        # Implementation here
        pass
'''
    
    @pytest.mark.asyncio
    async def test_modify_existing_file_success(self, modifier, mock_llm_client, sample_repository_map, 
                                               sample_analysis, sample_current_content):
        """Test successful file modification."""
        result = await modifier.modify_existing_file(
            file_path="app/services.py",
            current_content=sample_current_content,
            analysis=sample_analysis,
            repository_map=sample_repository_map,
            llm_client=mock_llm_client
        )
        
        assert result is not None
        assert "UserService" in result
        assert "authenticate_user" in result  # New method should be added
        
        # Verify LLM was called with proper context
        mock_llm_client.generate_response.assert_called_once()
        call_args = mock_llm_client.generate_response.call_args[0][0]
        assert len(call_args) == 2  # System and user messages
    
    @pytest.mark.asyncio
    async def test_modify_existing_file_with_no_target_classes(self, modifier, mock_llm_client, 
                                                              sample_repository_map, sample_current_content):
        """Test file modification when no target classes are found."""
        analysis = {
            "summary": "Add utility function",
            "files_to_modify": ["app/utils.py"],  # Different file
            "existing_classes_to_enhance": []
        }
        
        result = await modifier.modify_existing_file(
            file_path="app/services.py",
            current_content=sample_current_content,
            analysis=analysis,
            repository_map=sample_repository_map,
            llm_client=mock_llm_client
        )
        
        assert result is not None
        # Should still call LLM for general modification
        mock_llm_client.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_modify_existing_file_llm_failure(self, modifier, sample_repository_map, 
                                                   sample_analysis, sample_current_content):
        """Test handling of LLM failure."""
        mock_llm_client = AsyncMock()
        mock_llm_client.create_system_message.return_value = {"role": "system", "content": "test"}
        mock_llm_client.create_user_message.return_value = {"role": "user", "content": "test"}
        mock_llm_client.generate_response.side_effect = Exception("LLM API error")
        
        result = await modifier.modify_existing_file(
            file_path="app/services.py",
            current_content=sample_current_content,
            analysis=sample_analysis,
            repository_map=sample_repository_map,
            llm_client=mock_llm_client
        )
        
        assert result is None
    
    def test_modify_existing_class_success(self, modifier, sample_current_content):
        """Test successful class modification."""
        class_info = ClassInfo(
            name="UserService",
            file_path="app/services.py",
            methods=["create_user", "get_user"],
            dependencies=["db_client"],
            purpose="Service for managing users"
        )
        
        new_methods = [
            "async def authenticate_user(self, email, password):\n    \"\"\"Authenticate user.\"\"\"\n    pass"
        ]
        
        result = modifier._modify_existing_class(sample_current_content, class_info, new_methods)
        
        assert result is not None
        assert "authenticate_user" in result
        assert "class UserService:" in result
        # Should preserve existing methods
        assert "create_user" in result
        assert "get_user" in result
    
    def test_modify_existing_class_not_found(self, modifier, sample_current_content):
        """Test class modification when class is not found."""
        class_info = ClassInfo(
            name="NonExistentService",
            file_path="app/services.py",
            methods=[],
            dependencies=[],
            purpose="Non-existent service"
        )
        
        new_methods = ["def new_method(self): pass"]
        
        result = modifier._modify_existing_class(sample_current_content, class_info, new_methods)
        
        # Should return original content when class not found
        assert result == sample_current_content
    
    def test_enhance_existing_function_success(self, modifier):
        """Test successful function enhancement."""
        current_content = '''
def process_data(data):
    """Process some data."""
    return data.strip()
'''
        
        enhancement = "Add validation and error handling"
        
        result = modifier._enhance_existing_function(current_content, "process_data", enhancement)
        
        assert result is not None
        assert "process_data" in result
        # Should contain enhancement comment or logic
        assert len(result) > len(current_content)
    
    def test_enhance_existing_function_not_found(self, modifier):
        """Test function enhancement when function is not found."""
        current_content = '''
def existing_function():
    pass
'''
        
        result = modifier._enhance_existing_function(current_content, "non_existent_function", "enhancement")
        
        # Should return original content when function not found
        assert result == current_content
    
    def test_build_architecture_aware_prompt(self, modifier, sample_repository_map, sample_analysis):
        """Test building architecture-aware prompts."""
        file_classes = [sample_repository_map.classes["UserService"]]
        
        prompt = modifier._build_architecture_aware_prompt(
            sample_analysis, sample_repository_map, file_classes, "app/services.py"
        )
        
        assert "REPOSITORY ARCHITECTURE CONTEXT" in prompt
        assert "UserService" in prompt
        assert "Service Layer Pattern" in prompt
        assert "existing functionality" in prompt.lower()
    
    def test_build_architecture_aware_prompt_no_context(self, modifier, sample_analysis):
        """Test building prompts without repository context."""
        prompt = modifier._build_architecture_aware_prompt(
            sample_analysis, None, [], "app/services.py"
        )
        
        assert "Add user authentication functionality" in prompt
        # Should still work without repository context
        assert len(prompt) > 0
    
    def test_extract_class_methods(self, modifier):
        """Test extracting methods from class definition."""
        class_content = '''
class TestClass:
    def __init__(self):
        pass
    
    def method1(self):
        pass
    
    async def method2(self, param):
        return param
    
    @property
    def property1(self):
        return self._value
'''
        
        methods = modifier._extract_class_methods(class_content)
        
        assert len(methods) >= 3
        assert any("method1" in method for method in methods)
        assert any("method2" in method for method in methods)
        assert any("property1" in method for method in methods)
    
    def test_extract_class_methods_no_methods(self, modifier):
        """Test extracting methods from class with no methods."""
        class_content = '''
class EmptyClass:
    pass
'''
        
        methods = modifier._extract_class_methods(class_content)
        
        assert len(methods) == 0
    
    def test_insert_methods_into_class(self, modifier):
        """Test inserting methods into existing class."""
        class_content = '''
class TestClass:
    def __init__(self):
        self.value = 0
    
    def existing_method(self):
        return self.value
'''
        
        new_methods = [
            "    def new_method(self):\n        \"\"\"New method.\"\"\"\n        return True"
        ]
        
        result = modifier._insert_methods_into_class(class_content, new_methods)
        
        assert "new_method" in result
        assert "existing_method" in result
        assert "__init__" in result
        # Should maintain proper indentation
        assert "    def new_method" in result


@pytest.mark.integration
class TestCodeModifierIntegration:
    """Integration tests for CodeModifier."""
    
    @pytest.mark.asyncio
    async def test_full_modification_workflow(self):
        """Test the complete modification workflow."""
        modifier = CodeModifier()
        
        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.create_system_message.return_value = {"role": "system", "content": "test"}
        mock_llm.create_user_message.return_value = {"role": "user", "content": "test"}
        mock_llm.generate_response.return_value = '''
"""Enhanced user service module."""

import asyncio
from typing import Optional
import bcrypt


class UserService:
    """Service for managing users with authentication."""
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        """Create a new user."""
        # Existing implementation preserved
        pass
    
    async def get_user(self, user_id):
        """Get user by ID."""
        # Existing implementation preserved
        pass
    
    async def authenticate_user(self, email, password):
        """Authenticate user with email and password."""
        user = await self.get_user_by_email(email)
        if user and bcrypt.checkpw(password.encode(), user.password_hash):
            return user
        return None
    
    async def get_user_by_email(self, email):
        """Get user by email address."""
        return await self.db_client.find_one({"email": email})
'''
        
        # Create repository map
        repository_map = RepositoryMap(
            modules={},
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
        
        # Analysis result
        analysis = {
            "summary": "Add user authentication",
            "files_to_modify": ["app/services.py"],
            "existing_classes_to_enhance": ["UserService"],
            "requirements": ["Add authentication method"],
            "dependencies": ["bcrypt"]
        }
        
        # Current file content
        current_content = '''
"""User service module."""

import asyncio
from typing import Optional


class UserService:
    """Service for managing users."""
    
    def __init__(self, db_client):
        self.db_client = db_client
    
    async def create_user(self, user_data):
        """Create a new user."""
        pass
    
    async def get_user(self, user_id):
        """Get user by ID."""
        pass
'''
        
        # Execute modification
        result = await modifier.modify_existing_file(
            file_path="app/services.py",
            current_content=current_content,
            analysis=analysis,
            repository_map=repository_map,
            llm_client=mock_llm
        )
        
        # Verify results
        assert result is not None
        assert "authenticate_user" in result
        assert "bcrypt" in result
        assert "UserService" in result
        
        # Verify LLM was called properly
        mock_llm.generate_response.assert_called_once()