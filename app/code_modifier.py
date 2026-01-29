"""Intelligent Code Modification System for Architecture-Aware Changes."""

import ast
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .repository_analyzer import ClassInfo, FunctionInfo, RepositoryMap

logger = logging.getLogger(__name__)


@dataclass
class ModificationTarget:
    """Target for code modification."""
    file_path: str
    target_type: str  # 'class', 'function', 'module'
    target_name: str
    modification_type: str  # 'add_method', 'enhance_function', 'add_import', 'create_class'
    content: str
    line_number: Optional[int] = None


class CodeModifier:
    """Intelligent code modification system that understands existing architecture."""
    
    def __init__(self, repository_map: RepositoryMap):
        self.repository_map = repository_map
    
    def modify_existing_class(self, class_info: ClassInfo, new_methods: List[str], new_content: str) -> str:
        """Add methods to existing class while preserving structure."""
        try:
            logger.info(f"Modifying existing class {class_info.name} in {class_info.file_path}")
            
            # Parse the new content to understand what needs to be added
            modifications = self._parse_class_modifications(new_content, class_info.name)
            
            # Read current file content (this would be provided by the caller)
            # For now, we'll return the modification instructions
            return self._generate_class_modification_instructions(class_info, modifications)
            
        except Exception as e:
            logger.error(f"Failed to modify class {class_info.name}: {e}")
            return new_content  # Fallback to original content
    
    def enhance_existing_function(self, func_info: FunctionInfo, enhancement: str) -> str:
        """Enhance existing function while preserving existing logic."""
        try:
            logger.info(f"Enhancing function {func_info.name} in {func_info.file_path}")
            
            # Generate enhancement instructions
            return self._generate_function_enhancement_instructions(func_info, enhancement)
            
        except Exception as e:
            logger.error(f"Failed to enhance function {func_info.name}: {e}")
            return enhancement  # Fallback to original enhancement
    
    def determine_modification_strategy(self, requirement: str, suggested_files: List[str]) -> List[ModificationTarget]:
        """Determine the best modification strategy based on existing code."""
        targets = []
        
        try:
            # Find relevant existing classes
            relevant_classes = self.repository_map.find_relevant_classes(requirement)
            
            for class_info in relevant_classes:
                # Check if any suggested files match this class
                if class_info.file_path in suggested_files:
                    targets.append(ModificationTarget(
                        file_path=class_info.file_path,
                        target_type='class',
                        target_name=class_info.name,
                        modification_type='add_method',
                        content=f"Add functionality to existing {class_info.name} class",
                        line_number=class_info.line_number
                    ))
            
            # Check for files that don't have existing classes - these might need new classes
            for file_path in suggested_files:
                if not any(target.file_path == file_path for target in targets):
                    # Check if file exists in repository
                    if file_path in self.repository_map.modules:
                        targets.append(ModificationTarget(
                            file_path=file_path,
                            target_type='module',
                            target_name=file_path,
                            modification_type='add_function',
                            content=f"Add new functionality to existing module {file_path}"
                        ))
                    else:
                        targets.append(ModificationTarget(
                            file_path=file_path,
                            target_type='module',
                            target_name=file_path,
                            modification_type='create_file',
                            content=f"Create new file {file_path}"
                        ))
            
            logger.info(f"Determined {len(targets)} modification targets for requirement")
            return targets
            
        except Exception as e:
            logger.error(f"Failed to determine modification strategy: {e}")
            return []
    
    def _parse_class_modifications(self, new_content: str, class_name: str) -> Dict[str, List[str]]:
        """Parse new content to extract what should be added to existing class."""
        modifications = {
            'methods': [],
            'attributes': [],
            'imports': []
        }
        
        try:
            # Try to parse as Python code
            tree = ast.parse(new_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Extract methods from the new class definition
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_code = ast.get_source_segment(new_content, item)
                            if method_code:
                                modifications['methods'].append(method_code)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        modifications['imports'].append(f"import {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = [alias.name for alias in node.names]
                        modifications['imports'].append(f"from {node.module} import {', '.join(names)}")
            
        except SyntaxError:
            # If parsing fails, try to extract methods using regex
            method_pattern = r'def\s+(\w+)\s*\([^)]*\):[^:]*?(?=\n\s*def|\n\s*class|\Z)'
            methods = re.findall(method_pattern, new_content, re.DOTALL)
            modifications['methods'] = methods
        
        return modifications
    
    def _generate_class_modification_instructions(self, class_info: ClassInfo, modifications: Dict[str, List[str]]) -> str:
        """Generate instructions for modifying an existing class."""
        instructions = f"""# Modification instructions for class {class_info.name} in {class_info.file_path}

# EXISTING CLASS: {class_info.name}
# Location: {class_info.file_path}:{class_info.line_number}
# Purpose: {class_info.purpose or 'No description available'}

# EXISTING METHODS:
"""
        
        for method in class_info.methods:
            instructions += f"# - {method.name}({', '.join(method.args)})\n"
        
        instructions += "\n# NEW METHODS TO ADD:\n"
        for method in modifications.get('methods', []):
            instructions += f"# ADD: {method}\n"
        
        instructions += "\n# IMPORTS TO ADD:\n"
        for import_stmt in modifications.get('imports', []):
            instructions += f"# ADD: {import_stmt}\n"
        
        instructions += f"""
# MODIFICATION STRATEGY:
# 1. Preserve all existing methods in {class_info.name}
# 2. Add new methods while maintaining class structure
# 3. Add necessary imports at the top of the file
# 4. Ensure proper indentation and Python syntax
# 5. Add docstrings for new methods following existing patterns

# IMPORTANT: Do not remove or modify existing methods unless explicitly required.
# Only ADD new functionality to the existing class structure.
"""
        
        return instructions
    
    def _generate_function_enhancement_instructions(self, func_info: FunctionInfo, enhancement: str) -> str:
        """Generate instructions for enhancing an existing function."""
        instructions = f"""# Enhancement instructions for function {func_info.name} in {func_info.file_path}

# EXISTING FUNCTION: {func_info.name}
# Location: {func_info.file_path}:{func_info.line_number}
# Arguments: {', '.join(func_info.args)}
# Return Type: {func_info.return_type or 'Not specified'}
# Is Method: {func_info.is_method}
# Class: {func_info.class_name or 'N/A'}

# CURRENT DOCSTRING:
# {func_info.docstring or 'No docstring available'}

# ENHANCEMENT REQUEST:
# {enhancement}

# MODIFICATION STRATEGY:
# 1. Preserve existing function signature: {func_info.name}({', '.join(func_info.args)})
# 2. Maintain existing functionality - do not break current behavior
# 3. Add new functionality as specified in enhancement
# 4. Update docstring to reflect new capabilities
# 5. Add proper error handling for new functionality
# 6. Maintain existing return type: {func_info.return_type or 'inferred'}

# IMPORTANT: Enhance the function without breaking existing behavior.
# Add new capabilities while preserving all current functionality.
"""
        
        return instructions
    
    def generate_architecture_aware_prompt(self, requirement: str, target_files: List[str]) -> str:
        """Generate architecture-aware prompt for LLM."""
        prompt = f"""You are an expert software developer working with an existing codebase.

REQUIREMENT: {requirement}

REPOSITORY ARCHITECTURE CONTEXT:
{self.repository_map.get_structure_summary()}

ARCHITECTURAL PATTERNS IN USE:
{chr(10).join(f"- {pattern}" for pattern in self.repository_map.architecture_patterns)}

NAMING CONVENTIONS:
{chr(10).join(f"- {pattern}" for pattern in self.repository_map.get_naming_patterns())}

TARGET FILES FOR MODIFICATION: {', '.join(target_files)}

EXISTING CLASSES THAT MIGHT BE RELEVANT:
"""
        
        # Add relevant existing classes
        relevant_classes = self.repository_map.find_relevant_classes(requirement)
        for class_info in relevant_classes[:5]:  # Limit to top 5
            prompt += f"""
- {class_info.name} ({class_info.file_path}:{class_info.line_number})
  Purpose: {class_info.purpose or 'No description'}
  Methods: {', '.join(method.name for method in class_info.methods[:5])}
"""
        
        prompt += f"""
MODIFICATION RULES:
1. ALWAYS prefer modifying existing classes over creating new ones
2. If a class already handles similar functionality, ADD methods to that class
3. Follow existing naming conventions and architectural patterns
4. Preserve all existing functionality - never remove or break existing code
5. Add proper imports and maintain file structure
6. Use existing patterns for error handling, logging, and documentation

CRITICAL: If you see an existing class that handles similar functionality to what's requested,
you MUST modify that existing class instead of creating a new one.

For each target file, determine:
- Does it contain existing classes that should be modified?
- What new methods/functions need to be added?
- What imports are needed?
- How to maintain architectural consistency?

Provide the complete modified file content that preserves existing functionality while adding the new requirements.
"""
        
        return prompt
    
    def validate_modification(self, original_content: str, modified_content: str, class_name: str) -> Tuple[bool, List[str]]:
        """Validate that modification preserves existing functionality."""
        issues = []
        
        try:
            # Parse both versions
            original_tree = ast.parse(original_content)
            modified_tree = ast.parse(modified_content)
            
            # Find the target class in both versions
            original_class = None
            modified_class = None
            
            for node in ast.walk(original_tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    original_class = node
                    break
            
            for node in ast.walk(modified_tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    modified_class = node
                    break
            
            if not original_class:
                issues.append(f"Original class {class_name} not found")
                return False, issues
            
            if not modified_class:
                issues.append(f"Modified class {class_name} not found")
                return False, issues
            
            # Check that all original methods are preserved
            original_methods = {item.name for item in original_class.body if isinstance(item, ast.FunctionDef)}
            modified_methods = {item.name for item in modified_class.body if isinstance(item, ast.FunctionDef)}
            
            missing_methods = original_methods - modified_methods
            if missing_methods:
                issues.append(f"Missing original methods: {', '.join(missing_methods)}")
            
            # Check for syntax errors
            try:
                compile(modified_content, '<string>', 'exec')
            except SyntaxError as e:
                issues.append(f"Syntax error in modified code: {e}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation failed: {e}")
            return False, issues
    
    def suggest_file_organization(self, requirement: str) -> Dict[str, str]:
        """Suggest how to organize new functionality within existing file structure."""
        suggestions = {}
        
        # Analyze requirement to suggest appropriate files
        requirement_lower = requirement.lower()
        
        # Check for common patterns
        if 'agent' in requirement_lower:
            # Look for existing agent files
            agent_files = [path for path in self.repository_map.modules.keys() if 'agent' in path.lower()]
            if agent_files:
                suggestions['modify_existing'] = f"Consider modifying existing agent file: {agent_files[0]}"
        
        if 'client' in requirement_lower:
            client_files = [path for path in self.repository_map.modules.keys() if 'client' in path.lower()]
            if client_files:
                suggestions['modify_existing'] = f"Consider modifying existing client file: {client_files[0]}"
        
        if 'database' in requirement_lower or 'db' in requirement_lower:
            db_files = [path for path in self.repository_map.modules.keys() if 'database' in path.lower() or 'db' in path.lower()]
            if db_files:
                suggestions['modify_existing'] = f"Consider modifying existing database file: {db_files[0]}"
        
        # Suggest architectural consistency
        if self.repository_map.architecture_patterns:
            suggestions['follow_patterns'] = f"Follow existing patterns: {', '.join(self.repository_map.architecture_patterns)}"
        
        return suggestions