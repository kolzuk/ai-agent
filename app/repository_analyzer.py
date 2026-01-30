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
class ProjectProfile:
    """Information about the project's language, build system, and structure."""
    primary_language: str = "unknown"
    build_system: str = "unknown"
    build_files: List[str] = field(default_factory=list)
    source_directories: List[str] = field(default_factory=list)
    test_directories: List[str] = field(default_factory=list)
    language_files: Dict[str, int] = field(default_factory=dict)  # language -> file count
    project_type: str = "unknown"  # web, library, application, etc.


@dataclass
class RepositoryMap:
    """Complete map of repository structure and relationships."""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    architecture_patterns: List[str] = field(default_factory=list)
    project_profile: ProjectProfile = field(default_factory=ProjectProfile)
    
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
    
    async def analyze_repository(self, github_client, owner: str, repo: str) -> RepositoryMap:
        """Analyze repository structure using GitHub client."""
        try:
            logger.info(f"Starting repository structure analysis for {owner}/{repo}")
            
            # First, analyze project profile (build files, languages, structure)
            await self._analyze_project_profile(github_client, owner, repo)
            
            # Get all relevant files based on detected languages
            if self.repository_map.project_profile.primary_language == "scala":
                # For Scala projects, analyze Scala files
                scala_files = await self._get_scala_files(github_client, owner, repo)
                for file_path in scala_files:
                    try:
                        await self._analyze_scala_file(github_client, owner, repo, file_path)
                    except Exception as e:
                        logger.warning(f"Failed to analyze Scala file {file_path}: {e}")
            elif self.repository_map.project_profile.primary_language == "java":
                # For Java projects, analyze Java files
                java_files = await self._get_java_files(github_client, owner, repo)
                for file_path in java_files:
                    try:
                        await self._analyze_java_file(github_client, owner, repo, file_path)
                    except Exception as e:
                        logger.warning(f"Failed to analyze Java file {file_path}: {e}")
            
            # Always analyze Python files (for mixed projects or Python projects)
            python_files = await self._get_python_files(github_client, owner, repo)
            for file_path in python_files:
                try:
                    await self._analyze_file(github_client, owner, repo, file_path)
                except Exception as e:
                    logger.warning(f"Failed to analyze file {file_path}: {e}")
            
            # Build dependency graph
            self._build_dependency_graph()
            
            # Identify architectural patterns
            self._identify_patterns()
            
            logger.info(f"Repository analysis complete: {len(self.repository_map.classes)} classes, {len(self.repository_map.functions)} functions")
            logger.info(f"Project profile: {self.repository_map.project_profile.primary_language} project with {self.repository_map.project_profile.build_system} build system")
            return self.repository_map
            
        except Exception as e:
            logger.error(f"Repository analysis failed: {e}")
            return self.repository_map
    
    async def _get_python_files(self, github_client, owner: str, repo: str) -> List[str]:
        """Get list of all Python files in the repository."""
        try:
            all_files = await github_client.list_repository_files(owner, repo, "", "main")
            python_files = []
            for file_item in all_files:
                file_path = file_item["path"]
                if file_path.endswith('.py') and not file_path.startswith('.'):
                    python_files.append(file_path)
            
            logger.info(f"Found {len(python_files)} Python files")
            return python_files
            
        except Exception as e:
            logger.error(f"Failed to get Python files: {e}")
            return []
    
    async def _analyze_file(self, github_client, owner: str, repo: str, file_path: str) -> None:
        """Analyze a single Python file."""
        try:
            # Get file content
            file_data = await github_client.get_file_content(owner, repo, file_path, "main")
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
    
    def find_relevant_classes(self, repository_map: RepositoryMap, requirement: str) -> List[ClassInfo]:
        """Find classes relevant to a requirement using keyword matching."""
        return repository_map.find_relevant_classes(requirement)
    
    def suggest_modification_targets(self, repository_map: RepositoryMap, requirement: str) -> List[ClassInfo]:
        """Suggest which existing classes should be modified for a requirement."""
        return self.suggest_modification_target(requirement)
    
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
    
    async def _analyze_project_profile(self, github_client, owner: str, repo: str) -> None:
        """Analyze project profile including build files and language detection."""
        try:
            logger.info(f"Analyzing project profile for {owner}/{repo}")
            
            # Получить все файлы рекурсивно
            all_files = await github_client.list_repository_files(owner, repo, "", "main")
            
            # Инициализировать профиль проекта
            profile = ProjectProfile()
            
            # Анализировать build файлы и определить build систему
            build_files = []
            language_counts = {}
            source_dirs = set()
            test_dirs = set()
            
            for file_item in all_files:
                file_path = file_item["path"]
                
                # Анализ build файлов
                if file_path in ["build.sbt", "project/build.properties", "project/plugins.sbt"]:
                    build_files.append(file_path)
                    profile.build_system = "sbt"
                elif file_path in ["pom.xml"]:
                    build_files.append(file_path)
                    profile.build_system = "maven"
                elif file_path in ["build.gradle", "build.gradle.kts", "settings.gradle"]:
                    build_files.append(file_path)
                    profile.build_system = "gradle"
                elif file_path in ["package.json"]:
                    build_files.append(file_path)
                    profile.build_system = "npm"
                elif file_path in ["requirements.txt", "setup.py", "pyproject.toml"]:
                    build_files.append(file_path)
                    profile.build_system = "python"
                
                # Подсчет файлов по языкам
                if file_path.endswith('.scala'):
                    language_counts['scala'] = language_counts.get('scala', 0) + 1
                elif file_path.endswith('.java'):
                    language_counts['java'] = language_counts.get('java', 0) + 1
                elif file_path.endswith('.py'):
                    language_counts['python'] = language_counts.get('python', 0) + 1
                elif file_path.endswith('.js') or file_path.endswith('.ts'):
                    language_counts['javascript'] = language_counts.get('javascript', 0) + 1
                
                # Определение source и test директорий
                if '/src/main/' in file_path:
                    source_dirs.add(file_path.split('/src/main/')[0] + '/src/main')
                elif '/src/test/' in file_path:
                    test_dirs.add(file_path.split('/src/test/')[0] + '/src/test')
                elif file_path.startswith('src/'):
                    source_dirs.add('src')
                elif file_path.startswith('test/') or file_path.startswith('tests/'):
                    test_dirs.add(file_path.split('/')[0])
            
            profile.build_files = build_files
            profile.language_files = language_counts
            profile.source_directories = list(source_dirs)
            profile.test_directories = list(test_dirs)
            
            # Определить основной язык
            if language_counts:
                primary_lang = max(language_counts.items(), key=lambda x: x[1])[0]
                profile.primary_language = primary_lang
            
            # Определить тип проекта
            if "build.sbt" in build_files or language_counts.get('scala', 0) > 0:
                profile.project_type = "scala_application"
            elif "pom.xml" in build_files or language_counts.get('java', 0) > 0:
                profile.project_type = "java_application"
            elif any("app.py" in f["path"] or "main.py" in f["path"] for f in all_files):
                profile.project_type = "python_application"
            elif "package.json" in build_files:
                profile.project_type = "javascript_application"
            else:
                profile.project_type = "library"
            
            self.repository_map.project_profile = profile
            
            logger.info(f"Project profile detected: {profile.primary_language} {profile.project_type} with {profile.build_system}")
            logger.info(f"Language distribution: {language_counts}")
            logger.info(f"Build files: {build_files}")
            
        except Exception as e:
            logger.error(f"Failed to analyze project profile: {e}")
            # Установить профиль по умолчанию
            self.repository_map.project_profile = ProjectProfile()
    
    async def _get_scala_files(self, github_client, owner: str, repo: str) -> List[str]:
        """Get list of all Scala files in the repository."""
        try:
            all_files = await github_client.list_repository_files(owner, repo, "", "main")
            scala_files = []
            for file_item in all_files:
                # file_item теперь всегда dict
                file_path = file_item["path"]
                if file_path.endswith('.scala') and not file_path.startswith('.'):
                    scala_files.append(file_path)
            logger.info(f"Found {len(scala_files)} Scala files")
            return scala_files
        except Exception as e:
            logger.error(f"Failed to get Scala files: {e}")
            return []
    
    async def _get_java_files(self, github_client, owner: str, repo: str) -> List[str]:
        """Get list of all Java files in the repository."""
        try:
            all_files = await github_client.list_repository_files(owner, repo, "", "main")
            java_files = []
            for file_item in all_files:
                file_path = file_item["path"]
                if file_path.endswith('.java') and not file_path.startswith('.'):
                    java_files.append(file_path)
            logger.info(f"Found {len(java_files)} Java files")
            return java_files
        except Exception as e:
            logger.error(f"Failed to get Java files: {e}")
            return []
    
    async def _analyze_scala_file(self, github_client, owner: str, repo: str, file_path: str) -> None:
        """Analyze a Scala file (basic analysis without AST parsing)."""
        try:
            # Get file content
            file_data = await github_client.get_file_content(owner, repo, file_path, "main")
            if not file_data:
                return
            
            import base64
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            
            # Basic Scala parsing (without full AST)
            lines = content.split('\n')
            
            # Extract classes, objects, and traits
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Match class definitions
                if line.startswith('class ') or line.startswith('case class '):
                    class_name = self._extract_scala_class_name(line)
                    if class_name:
                        class_info = ClassInfo(
                            name=class_name,
                            file_path=file_path,
                            line_number=i + 1,
                            docstring=self._extract_scala_docstring(lines, i)
                        )
                        self.repository_map.classes[class_name] = class_info
                
                # Match object definitions
                elif line.startswith('object '):
                    object_name = self._extract_scala_object_name(line)
                    if object_name:
                        class_info = ClassInfo(
                            name=object_name,
                            file_path=file_path,
                            line_number=i + 1,
                            docstring=self._extract_scala_docstring(lines, i)
                        )
                        self.repository_map.classes[object_name] = class_info
                
                # Match trait definitions
                elif line.startswith('trait '):
                    trait_name = self._extract_scala_trait_name(line)
                    if trait_name:
                        class_info = ClassInfo(
                            name=trait_name,
                            file_path=file_path,
                            line_number=i + 1,
                            docstring=self._extract_scala_docstring(lines, i)
                        )
                        self.repository_map.classes[trait_name] = class_info
            
            logger.debug(f"Analyzed Scala file: {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to analyze Scala file {file_path}: {e}")
    
    async def _analyze_java_file(self, github_client, owner: str, repo: str, file_path: str) -> None:
        """Analyze a Java file (basic analysis without AST parsing)."""
        try:
            # Get file content
            file_data = await github_client.get_file_content(owner, repo, file_path, "main")
            if not file_data:
                return
            
            import base64
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            
            # Basic Java parsing (without full AST)
            lines = content.split('\n')
            
            # Extract classes and interfaces
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Match class definitions
                if 'class ' in line and not line.startswith('//'):
                    class_name = self._extract_java_class_name(line)
                    if class_name:
                        class_info = ClassInfo(
                            name=class_name,
                            file_path=file_path,
                            line_number=i + 1,
                            docstring=self._extract_java_docstring(lines, i)
                        )
                        self.repository_map.classes[class_name] = class_info
                
                # Match interface definitions
                elif 'interface ' in line and not line.startswith('//'):
                    interface_name = self._extract_java_interface_name(line)
                    if interface_name:
                        class_info = ClassInfo(
                            name=interface_name,
                            file_path=file_path,
                            line_number=i + 1,
                            docstring=self._extract_java_docstring(lines, i)
                        )
                        self.repository_map.classes[interface_name] = class_info
            
            logger.debug(f"Analyzed Java file: {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to analyze Java file {file_path}: {e}")
    
    def _extract_scala_class_name(self, line: str) -> Optional[str]:
        """Extract class name from Scala class definition line."""
        try:
            # Handle: class ClassName, case class ClassName
            if 'class ' in line:
                parts = line.split('class ')[1].split()
                if parts:
                    name = parts[0].split('(')[0].split('[')[0]  # Remove generics and constructor params
                    return name if name.isidentifier() else None
        except:
            pass
        return None
    
    def _extract_scala_object_name(self, line: str) -> Optional[str]:
        """Extract object name from Scala object definition line."""
        try:
            if 'object ' in line:
                parts = line.split('object ')[1].split()
                if parts:
                    name = parts[0].split('(')[0].split('[')[0]
                    return name if name.isidentifier() else None
        except:
            pass
        return None
    
    def _extract_scala_trait_name(self, line: str) -> Optional[str]:
        """Extract trait name from Scala trait definition line."""
        try:
            if 'trait ' in line:
                parts = line.split('trait ')[1].split()
                if parts:
                    name = parts[0].split('(')[0].split('[')[0]
                    return name if name.isidentifier() else None
        except:
            pass
        return None
    
    def _extract_java_class_name(self, line: str) -> Optional[str]:
        """Extract class name from Java class definition line."""
        try:
            # Handle: public class ClassName, class ClassName
            if 'class ' in line:
                parts = line.split('class ')[1].split()
                if parts:
                    name = parts[0].split('<')[0].split('{')[0].strip()  # Remove generics and opening brace
                    return name if name.isidentifier() else None
        except:
            pass
        return None
    
    def _extract_java_interface_name(self, line: str) -> Optional[str]:
        """Extract interface name from Java interface definition line."""
        try:
            if 'interface ' in line:
                parts = line.split('interface ')[1].split()
                if parts:
                    name = parts[0].split('<')[0].split('{')[0].strip()
                    return name if name.isidentifier() else None
        except:
            pass
        return None
    
    def _extract_scala_docstring(self, lines: List[str], class_line: int) -> Optional[str]:
        """Extract Scala docstring (/** ... */) before class definition."""
        try:
            # Look backwards for /** comment
            for i in range(class_line - 1, max(0, class_line - 10), -1):
                line = lines[i].strip()
                if line.endswith('*/'):
                    # Found end of comment, collect the comment
                    comment_lines = []
                    for j in range(i, -1, -1):
                        comment_line = lines[j].strip()
                        comment_lines.insert(0, comment_line)
                        if comment_line.startswith('/**'):
                            # Found start of comment
                            comment = '\n'.join(comment_lines)
                            # Clean up comment
                            comment = comment.replace('/**', '').replace('*/', '').replace('*', '').strip()
                            return comment if comment else None
        except:
            pass
        return None
    
    def _extract_java_docstring(self, lines: List[str], class_line: int) -> Optional[str]:
        """Extract Java docstring (/** ... */) before class definition."""
        return self._extract_scala_docstring(lines, class_line)  # Same format

    async def build_file_index(self, github_client, owner: str, repo: str) -> Dict[str, List[str]]:
        """Построить рекурсивный индекс всех файлов в репозитории."""
        try:
            # Получить все файлы рекурсивно (теперь работает правильно!)
            all_files = await github_client.list_repository_files(owner, repo, "", "main")
            
            file_index = {
                "scala": [],
                "java": [],
                "python": [],
                "all": []
            }
            
            for file_item in all_files:
                # Теперь file_item всегда dict с полной информацией о файле
                file_path = file_item["path"]
                file_index["all"].append(file_path)
                
                # Индексировать по расширениям
                if file_path.endswith('.scala'):
                    file_index["scala"].append(file_path)
                elif file_path.endswith('.java'):
                    file_index["java"].append(file_path)
                elif file_path.endswith('.py'):
                    file_index["python"].append(file_path)
            
            logger.info(f"File index built: {len(file_index['scala'])} Scala, {len(file_index['java'])} Java, {len(file_index['python'])} Python files")
            return file_index
            
        except Exception as e:
            logger.error(f"Failed to build file index: {e}")
            return {"scala": [], "java": [], "python": [], "all": []}

    def find_files_by_pattern(self, file_index: Dict[str, List[str]], pattern: str, language: str = None) -> List[str]:
        """Найти файлы по паттерну имени."""
        pattern_lower = pattern.lower()
        candidates = []
        
        # Выбрать файлы для поиска
        if language:
            search_files = file_index.get(language, [])
        else:
            search_files = file_index.get("all", [])
        
        for file_path in search_files:
            filename = file_path.split('/')[-1]
            name_without_ext = filename.split('.')[0]
            
            # Точное совпадение имени файла
            if name_without_ext.lower() == pattern_lower:
                candidates.append(file_path)
            # Частичное совпадение
            elif pattern_lower in name_without_ext.lower():
                candidates.append(file_path)
        
        return candidates

    async def find_target_files_for_entity(self, github_client, owner: str, repo: str, entity_name: str) -> Dict[str, Any]:
        """Найти целевые файлы для сущности (класса/объекта)."""
        try:
            logger.info(f"=== SEARCHING FOR ENTITY: {entity_name} ===")
            
            # Построить индекс файлов
            file_index = await self.build_file_index(github_client, owner, repo)
            
            # Логировать статистику файлов
            logger.info(f"File index statistics:")
            for lang, files in file_index.items():
                logger.info(f"  - {lang}: {len(files)} files")
                if files and lang != "all":
                    logger.info(f"    Examples: {files[:3]}")
            
            # Определить основной язык проекта
            primary_language = self.repository_map.project_profile.primary_language
            logger.info(f"Primary language detected: {primary_language}")
            
            candidates = {
                "exact_matches": [],
                "partial_matches": [],
                "content_matches": [],
                "primary_language": primary_language
            }
            
            # 1. Поиск по имени файла
            logger.info(f"Searching for files matching '{entity_name}' in {primary_language} files...")
            exact_files = self.find_files_by_pattern(file_index, entity_name, primary_language)
            candidates["exact_matches"] = exact_files
            logger.info(f"Found {len(exact_files)} exact matches: {exact_files}")
            
            # 2. Поиск по содержимому (только для небольших репозиториев)
            lang_files = file_index.get(primary_language, [])
            if len(lang_files) < 50:  # Лимит для производительности
                logger.info(f"Searching in content of {len(lang_files)} {primary_language} files...")
                content_matches = await self._search_entity_in_content(
                    github_client, owner, repo, entity_name, lang_files
                )
                candidates["content_matches"] = content_matches
                logger.info(f"Found {len(content_matches)} content matches: {content_matches}")
            else:
                logger.info(f"Skipping content search: too many files ({len(lang_files)})")
            
            # 3. Логирование результатов
            logger.info(f"=== ENTITY SEARCH RESULTS FOR '{entity_name}' ===")
            logger.info(f"  - Exact matches: {len(candidates['exact_matches'])}")
            logger.info(f"  - Content matches: {len(candidates['content_matches'])}")
            
            # 4. Выбор лучшего файла
            best_file = self.select_best_target_file(candidates, entity_name)
            if best_file:
                logger.info(f"✅ SELECTED TARGET FILE: {best_file}")
            else:
                logger.warning(f"❌ NO TARGET FILE FOUND FOR: {entity_name}")
                logger.warning(f"Available {primary_language} files: {lang_files[:10]}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to find target files for entity {entity_name}: {e}")
            return {"exact_matches": [], "partial_matches": [], "content_matches": [], "primary_language": "unknown"}

    async def _search_entity_in_content(self, github_client, owner: str, repo: str, entity_name: str, file_paths: List[str]) -> List[str]:
        """Поиск сущности в содержимом файлов."""
        matches = []
        
        for file_path in file_paths[:20]:  # Лимит для производительности
            try:
                file_data = await github_client.get_file_content(owner, repo, file_path, "main")
                if file_data:
                    import base64
                    content = base64.b64decode(file_data["content"]).decode('utf-8')
                    
                    # Поиск определений класса/объекта/трейта
                    if any(pattern in content for pattern in [
                        f"class {entity_name}",
                        f"object {entity_name}",
                        f"trait {entity_name}",
                        f"case class {entity_name}"
                    ]):
                        matches.append(file_path)
                        logger.info(f"Found {entity_name} definition in {file_path}")
                        
            except Exception as e:
                logger.debug(f"Could not search in {file_path}: {e}")
        
        return matches

    def select_best_target_file(self, candidates: Dict[str, Any], entity_name: str) -> Optional[str]:
        """Выбрать лучший целевой файл из кандидатов."""
        # Приоритет: точные совпадения > совпадения в содержимом
        if candidates["exact_matches"]:
            # Если есть точные совпадения, выбрать первый
            best_match = candidates["exact_matches"][0]
            logger.info(f"Selected exact match: {best_match}")
            return best_match
        
        if candidates["content_matches"]:
            # Если есть совпадения в содержимом, выбрать первый
            best_match = candidates["content_matches"][0]
            logger.info(f"Selected content match: {best_match}")
            return best_match
        
        logger.warning(f"No target file found for entity {entity_name}")
        return None