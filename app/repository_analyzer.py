"""Repository Structure Analysis System for Architecture-Aware Code Generation."""

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function in the codebase."""
    name: str
    file_path: str
    line_number: int
    args: List[str]
    docstring: Optional[str]
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    return_type: Optional[str] = None


@dataclass
class ClassInfo:
    """Information about a class in the codebase."""
    name: str
    file_path: str
    line_number: int
    methods: List[FunctionInfo] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    purpose: Optional[str] = None  # Extracted from docstring analysis


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    file_path: str
    imports: List[str] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    docstring: Optional[str] = None


@dataclass
class RepositoryMap:
    """Complete map of repository structure and relationships."""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    architecture_patterns: List[str] = field(default_factory=list)
    
    def get_main_classes(self) -> List[str]:
        """Get list of main classes in the repository."""
        return list(self.classes.keys())
    
    def find_relevant_classes(self, requirement: str) -> List[ClassInfo]:
        """Find classes relevant to a requirement using keyword matching."""
        relevant = []
        requirement_lower = requirement.lower()
        
        for class_info in self.classes.values():
            # Check class name similarity
            if any(word in class_info.name.lower() for word in requirement_lower.split()):
                relevant.append(class_info)
                continue
            
            # Check docstring similarity
            if class_info.docstring and any(word in class_info.docstring.lower() for word in requirement_lower.split()):
                relevant.append(class_info)
                continue
            
            # Check method names
            if any(word in method.name.lower() for method in class_info.methods for word in requirement_lower.split()):
                relevant.append(class_info)
        
        return relevant
    
    def get_structure_summary(self) -> str:
        """Get a summary of repository structure."""
        summary = f"Repository contains {len(self.modules)} modules, {len(self.classes)} classes, {len(self.functions)} functions\n"
        
        # Add main modules
        main_modules = [path for path in self.modules.keys() if not path.startswith('tests/')][:10]
        summary += f"Main modules: {', '.join(main_modules)}\n"
        
        # Add main classes
        main_classes = list(self.classes.keys())[:10]
        summary += f"Main classes: {', '.join(main_classes)}"
        
        return summary
    
    def get_naming_patterns(self) -> List[str]:
        """Extract naming patterns from the codebase."""
        patterns = []
        
        # Analyze class naming patterns
        class_names = list(self.classes.keys())
        if any('Agent' in name for name in class_names):
            patterns.append("Classes ending with 'Agent' for agent-like functionality")
        if any('Client' in name for name in class_names):
            patterns.append("Classes ending with 'Client' for API clients")
        if any('Manager' in name for name in class_names):
            patterns.append("Classes ending with 'Manager' for management functionality")
        
        return patterns
    
    def get_patterns(self) -> List[str]:
        """Get architectural patterns used in the codebase."""
        return self.architecture_patterns


class RepositoryAnalyzer:
    """Analyzes repository structure to understand codebase architecture."""
    
    def __init__(self):
        self.repository_map = RepositoryMap()
    
    async def analyze_repository(self, github_client) -> RepositoryMap:
        """Analyze repository structure using GitHub client."""
        try:
            logger.info("Starting repository structure analysis")
            
            # Get all Python files
            python_files = await self._get_python_files(github_client)
            
            # Analyze each file
            for file_path in python_files:
                try:
                    await self._analyze_file(github_client, file_path)
                except Exception as e:
                    logger.warning(f"Failed to analyze file {file_path}: {e}")
            
            # Build dependency graph
            self._build_dependency_graph()
            
            # Identify architectural patterns
            self._identify_patterns()
            
            logger.info(f"Repository analysis complete: {len(self.repository_map.classes)} classes, {len(self.repository_map.functions)} functions")
            return self.repository_map
            
        except Exception as e:
            logger.error(f"Repository analysis failed: {e}")
            return self.repository_map
    
    async def _get_python_files(self, github_client) -> List[str]:
        """Get list of all Python files in the repository."""
        try:
            # Get repository contents recursively
            all_files = await github_client.list_repository_files("", "main")
            python_files = [f for f in all_files if f.endswith('.py') and not f.startswith('.')]
            
            logger.info(f"Found {len(python_files)} Python files")
            return python_files
            
        except Exception as e:
            logger.error(f"Failed to get Python files: {e}")
            return []
    
    async def _analyze_file(self, github_client, file_path: str) -> None:
        """Analyze a single Python file."""
        try:
            # Get file content
            file_data = await github_client.get_file_content("", "", file_path, "main")
            if not file_data:
                return
            
            import base64
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            
            # Parse with AST
            tree = ast.parse(content)
            
            # Create module info
            module_info = ModuleInfo(file_path=file_path)
            
            # Extract module docstring
            if ast.get_docstring(tree):
                module_info.docstring = ast.get_docstring(tree)
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info.imports.append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info.imports.append(node.module)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, file_path)
                    module_info.classes.append(class_info)
                    self.repository_map.classes[class_info.name] = class_info
                
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                        func_info = self._extract_function_info(node, file_path)
                        module_info.functions.append(func_info)
                        self.repository_map.functions[f"{file_path}:{func_info.name}"] = func_info
            
            self.repository_map.modules[file_path] = module_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze file {file_path}: {e}")
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        class_info = ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node)
        )
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info.base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info.base_classes.append(f"{base.value.id}.{base.attr}")
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_info.decorators.append(decorator.id)
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, file_path, is_method=True, class_name=node.name)
                class_info.methods.append(method_info)
        
        # Extract purpose from docstring
        if class_info.docstring:
            class_info.purpose = self._extract_purpose_from_docstring(class_info.docstring)
        
        return class_info
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: str, is_method: bool = False, class_name: Optional[str] = None) -> FunctionInfo:
        """Extract information from a function AST node."""
        func_info = FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            is_method=is_method,
            class_name=class_name
        )
        
        # Extract arguments
        for arg in node.args.args:
            func_info.args.append(arg.arg)
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_info.decorators.append(decorator.id)
        
        # Extract return type annotation
        if node.returns:
            if isinstance(node.returns, ast.Name):
                func_info.return_type = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                func_info.return_type = str(node.returns.value)
        
        return func_info
    
    def _extract_purpose_from_docstring(self, docstring: str) -> str:
        """Extract purpose from docstring."""
        if not docstring:
            return ""
        
        # Take first sentence as purpose
        sentences = docstring.split('.')
        if sentences:
            return sentences[0].strip()
        
        return docstring[:100].strip()
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph between modules."""
        for module_path, module_info in self.repository_map.modules.items():
            dependencies = set()
            
            for import_name in module_info.imports:
                # Check if import is from current repository
                if any(import_name in other_path for other_path in self.repository_map.modules.keys()):
                    dependencies.add(import_name)
            
            self.repository_map.dependencies[module_path] = dependencies
    
    def _identify_patterns(self) -> None:
        """Identify architectural patterns in the codebase."""
        patterns = []
        
        # Check for common patterns
        class_names = list(self.repository_map.classes.keys())
        
        if any('Agent' in name for name in class_names):
            patterns.append("Agent Pattern - Classes for autonomous behavior")
        
        if any('Client' in name for name in class_names):
            patterns.append("Client Pattern - Classes for external API interaction")
        
        if any('Manager' in name for name in class_names):
            patterns.append("Manager Pattern - Classes for resource management")
        
        if any('Orchestrator' in name for name in class_names):
            patterns.append("Orchestrator Pattern - Classes for workflow coordination")
        
        # Check for FastAPI pattern
        if any('router' in module.file_path.lower() for module in self.repository_map.modules.values()):
            patterns.append("FastAPI Router Pattern - Modular API endpoints")
        
        # Check for database pattern
        if any('database' in module.file_path.lower() for module in self.repository_map.modules.values()):
            patterns.append("Database Layer Pattern - Separated data access")
        
        self.repository_map.architecture_patterns = patterns
    
    def find_existing_class(self, class_name: str) -> Optional[ClassInfo]:
        """Find existing class by name or similar functionality."""
        # Exact match
        if class_name in self.repository_map.classes:
            return self.repository_map.classes[class_name]
        
        # Fuzzy match
        class_name_lower = class_name.lower()
        for name, class_info in self.repository_map.classes.items():
            if class_name_lower in name.lower() or name.lower() in class_name_lower:
                return class_info
        
        return None
    
    def suggest_modification_target(self, requirement: str) -> List[ClassInfo]:
        """Suggest which existing classes should be modified for a requirement."""
        relevant_classes = self.repository_map.find_relevant_classes(requirement)
        
        # Sort by relevance (simple scoring)
        scored_classes = []
        requirement_words = set(requirement.lower().split())
        
        for class_info in relevant_classes:
            score = 0
            
            # Score based on class name
            class_words = set(class_info.name.lower().split())
            score += len(requirement_words.intersection(class_words)) * 3
            
            # Score based on docstring
            if class_info.docstring:
                docstring_words = set(class_info.docstring.lower().split())
                score += len(requirement_words.intersection(docstring_words))
            
            # Score based on method names
            for method in class_info.methods:
                method_words = set(method.name.lower().split())
                score += len(requirement_words.intersection(method_words)) * 2
            
            scored_classes.append((score, class_info))
        
        # Return sorted by score (highest first)
        scored_classes.sort(key=lambda x: x[0], reverse=True)
        return [class_info for score, class_info in scored_classes[:5]]  # Top 5 suggestions