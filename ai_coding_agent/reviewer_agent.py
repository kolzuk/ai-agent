"""AI Reviewer Agent for analyzing pull requests and providing feedback."""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from .github_client import GitHubClient
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Agent responsible for reviewing pull requests and providing feedback."""

    def __init__(self, github_client: GitHubClient, llm_client: LLMClient) -> None:
        """Initialize the Reviewer Agent."""
        self.github_client = github_client
        self.llm_client = llm_client

    async def review_pull_request(self, pr_number: int) -> Dict[str, any]:
        """Review a pull request and provide comprehensive feedback."""
        try:
            logger.info(f"Starting review of pull request #{pr_number}")
            
            # Get PR details
            pr = self.github_client.get_pull_request(pr_number)
            
            # Get related issue if exists
            issue_number = self._extract_issue_number(pr.title, pr.body or "")
            issue = None
            if issue_number:
                try:
                    issue = self.github_client.get_issue(issue_number)
                except Exception as e:
                    logger.warning(f"Could not fetch issue #{issue_number}: {e}")

            # Get PR files and changes
            pr_files = self.github_client.get_pr_files(pr_number)
            
            # Perform comprehensive review
            review_result = await self._perform_comprehensive_review(
                pr, issue, pr_files
            )
            
            # Post review results
            await self._post_review_results(pr_number, review_result)
            
            logger.info(f"Completed review of pull request #{pr_number}")
            return review_result

        except Exception as e:
            logger.error(f"Error reviewing pull request {pr_number}: {e}")
            return {"status": "error", "message": str(e)}

    async def _perform_comprehensive_review(
        self, pr, issue, pr_files: List[Dict]
    ) -> Dict[str, any]:
        """Perform a comprehensive review of the pull request."""
        try:
            # Analyze code quality
            code_quality_result = await self._analyze_code_quality(pr_files)
            
            # Check requirements compliance
            requirements_result = await self._check_requirements_compliance(
                pr, issue, pr_files
            )
            
            # Analyze security and best practices
            security_result = await self._analyze_security_and_practices(pr_files)
            
            # Generate overall assessment
            overall_assessment = await self._generate_overall_assessment(
                code_quality_result,
                requirements_result,
                security_result,
                pr,
                issue
            )
            
            return {
                "status": "completed",
                "overall_assessment": overall_assessment,
                "code_quality": code_quality_result,
                "requirements_compliance": requirements_result,
                "security_analysis": security_result,
                "recommendation": overall_assessment.get("recommendation", "needs_work"),
                "score": overall_assessment.get("score", 0)
            }

        except Exception as e:
            logger.error(f"Error performing comprehensive review: {e}")
            return {"status": "error", "message": str(e)}

    async def _analyze_code_quality(self, pr_files: List[Dict]) -> Dict[str, any]:
        """Analyze code quality of the changes."""
        try:
            system_prompt = """You are an expert code reviewer focusing on code quality.
            
            Analyze the provided code changes and evaluate:
            1. Code structure and organization
            2. Naming conventions
            3. Function/class design
            4. Error handling
            5. Documentation and comments
            6. Type hints usage
            7. Code complexity
            8. Potential bugs or issues
            
            Provide a JSON response with:
            {
                "score": 0-100,
                "issues": [{"type": "error|warning|info", "message": "description", "file": "filename", "line": number}],
                "strengths": ["list of positive aspects"],
                "suggestions": ["list of improvement suggestions"],
                "summary": "overall code quality assessment"
            }"""

            # Prepare code changes for analysis
            code_changes = []
            for file_info in pr_files:
                if file_info.get("patch"):
                    code_changes.append(f"File: {file_info['filename']}\n{file_info['patch']}")

            user_prompt = f"""Please analyze the following code changes:

{chr(10).join(code_changes[:10])}  # Limit to first 10 files to avoid token limits

Provide your code quality analysis."""

            messages = [
                self.llm_client.create_system_message(system_prompt),
                self.llm_client.create_user_message(user_prompt)
            ]
            
            response = await self.llm_client.generate_response(messages)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"score": 50, "summary": "Could not parse analysis", "issues": []}

        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {"score": 0, "summary": f"Analysis failed: {str(e)}", "issues": []}

    async def _check_requirements_compliance(
        self, pr, issue, pr_files: List[Dict]
    ) -> Dict[str, any]:
        """Check if the PR meets the requirements from the issue."""
        try:
            if not issue:
                return {
                    "score": 50,
                    "summary": "No related issue found for requirements check",
                    "compliance_items": []
                }

            system_prompt = """You are an expert at verifying if code changes meet specified requirements.
            
            Compare the issue requirements with the implemented changes and evaluate:
            1. Are all requirements addressed?
            2. Is the implementation correct?
            3. Are there any missing features?
            4. Does the solution match the expected approach?
            
            Provide a JSON response with:
            {
                "score": 0-100,
                "compliance_items": [{"requirement": "description", "status": "met|partial|missing", "notes": "explanation"}],
                "missing_features": ["list of missing features"],
                "summary": "overall compliance assessment"
            }"""

            # Prepare analysis data
            # Handle both dict and object formats for issue
            if hasattr(issue, 'title'):
                issue_title = issue.title
                issue_body = issue.body
            else:
                issue_title = issue.get('title', 'No title')
                issue_body = issue.get('body', 'No description')
            
            # Handle both dict and object formats for pr
            if hasattr(pr, 'title'):
                pr_title = pr.title
                pr_body = pr.body
            else:
                pr_title = pr.get('title', 'No title')
                pr_body = pr.get('body', 'No description')
            
            issue_content = f"Title: {issue_title}\nDescription: {issue_body or 'No description'}"
            pr_content = f"Title: {pr_title}\nDescription: {pr_body or 'No description'}"
            
            changed_files = [f"- {f['filename']} ({f['status']})" for f in pr_files]
            
            user_prompt = f"""Issue Requirements:
{issue_content}

Pull Request Implementation:
{pr_content}

Files Changed:
{chr(10).join(changed_files)}

Please analyze if the implementation meets the requirements."""

            messages = [
                self.llm_client.create_system_message(system_prompt),
                self.llm_client.create_user_message(user_prompt)
            ]
            
            response = await self.llm_client.generate_response(messages)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"score": 50, "summary": "Could not parse compliance analysis"}

        except Exception as e:
            logger.error(f"Error checking requirements compliance: {e}")
            return {"score": 0, "summary": f"Compliance check failed: {str(e)}"}

    async def _analyze_security_and_practices(self, pr_files: List[Dict]) -> Dict[str, any]:
        """Analyze security issues and best practices."""
        try:
            system_prompt = """You are a security expert and best practices reviewer.
            
            Analyze the code changes for:
            1. Security vulnerabilities
            2. Input validation
            3. Error handling security
            4. Dependency security
            5. Best practices compliance
            6. Performance considerations
            7. Maintainability issues
            
            Provide a JSON response with:
            {
                "score": 0-100,
                "security_issues": [{"severity": "high|medium|low", "issue": "description", "file": "filename"}],
                "best_practices": [{"type": "good|bad", "practice": "description", "file": "filename"}],
                "performance_notes": ["list of performance observations"],
                "summary": "overall security and practices assessment"
            }"""

            # Prepare code for analysis (focus on Python files)
            python_files = [f for f in pr_files if f['filename'].endswith('.py')]
            code_snippets = []
            
            for file_info in python_files[:5]:  # Limit to 5 files
                if file_info.get("patch"):
                    code_snippets.append(f"File: {file_info['filename']}\n{file_info['patch']}")

            if not code_snippets:
                return {
                    "score": 80,
                    "summary": "No source files to analyze",
                    "security_issues": [],
                    "best_practices": []
                }

            user_prompt = f"""Please analyze the following code changes for security and best practices:

{chr(10).join(code_snippets)}

Provide your security and best practices analysis."""

            messages = [
                self.llm_client.create_system_message(system_prompt),
                self.llm_client.create_user_message(user_prompt)
            ]
            
            response = await self.llm_client.generate_response(messages)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"score": 50, "summary": "Could not parse security analysis"}

        except Exception as e:
            logger.error(f"Error analyzing security and practices: {e}")
            return {"score": 0, "summary": f"Security analysis failed: {str(e)}"}

    async def _generate_overall_assessment(
        self,
        code_quality: Dict,
        requirements: Dict,
        security: Dict,
        pr,
        issue
    ) -> Dict[str, any]:
        """Generate an overall assessment and recommendation."""
        try:
            # Calculate weighted score
            code_score = code_quality.get("score", 0)
            req_score = requirements.get("score", 0)
            sec_score = security.get("score", 0)
            
            # Weighted average: code quality 40%, requirements 40%, security 20%
            overall_score = (code_score * 0.4 + req_score * 0.4 + sec_score * 0.2)
            
            # Determine recommendation
            if overall_score >= 85:
                recommendation = "approve"
                status = "✅ Ready to merge"
            elif overall_score >= 70:
                recommendation = "approve_with_suggestions"
                status = "⚠️ Approve with minor suggestions"
            elif overall_score >= 50:
                recommendation = "request_changes"
                status = "🔄 Changes requested"
            else:
                recommendation = "reject"
                status = "❌ Significant issues found"

            # Count critical issues
            critical_issues = 0
            for analysis in [code_quality, security]:
                issues = analysis.get("issues", []) + analysis.get("security_issues", [])
                critical_issues += len([i for i in issues if i.get("severity") == "high" or i.get("type") == "error"])

            return {
                "score": round(overall_score, 1),
                "recommendation": recommendation,
                "status": status,
                "critical_issues_count": critical_issues,
                "summary": f"Overall score: {overall_score:.1f}/100. {status}",
                "breakdown": {
                    "code_quality": code_score,
                    "requirements_compliance": req_score,
                    "security_and_practices": sec_score
                }
            }

        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return {
                "score": 0,
                "recommendation": "error",
                "status": "❌ Review failed",
                "summary": f"Assessment failed: {str(e)}"
            }

    async def _post_review_results(self, pr_number: int, review_result: Dict) -> None:
        """Post review results as comments on the pull request."""
        try:
            # Generate main review comment
            comment = self._format_review_comment(review_result)
            self.github_client.add_comment_to_pr(pr_number, comment)
            
            # If there are critical issues, add individual comments for each
            if review_result.get("code_quality", {}).get("issues"):
                issues_comment = self._format_issues_comment(
                    review_result["code_quality"]["issues"]
                )
                if issues_comment:
                    self.github_client.add_comment_to_pr(pr_number, issues_comment)

        except Exception as e:
            logger.error(f"Error posting review results: {e}")

    def _format_review_comment(self, review_result: Dict) -> str:
        """Format the main review comment."""
        overall = review_result.get("overall_assessment", {})
        code_quality = review_result.get("code_quality", {})
        requirements = review_result.get("requirements_compliance", {})
        security = review_result.get("security_analysis", {})
        
        comment = f"""## 🤖 AI Code Review

### {overall.get('status', 'Review Completed')}

**Overall Score:** {overall.get('score', 0)}/100

### 📊 Breakdown:
- **Code Quality:** {overall.get('breakdown', {}).get('code_quality', 0)}/100
- **Requirements Compliance:** {overall.get('breakdown', {}).get('requirements_compliance', 0)}/100  
- **Security & Best Practices:** {overall.get('breakdown', {}).get('security_and_practices', 0)}/100

### 📝 Summary:
{overall.get('summary', 'No summary available')}

"""

        # Add code quality details
        if code_quality.get("summary"):
            comment += f"""### 🔍 Code Quality Analysis:
{code_quality['summary']}

"""

        # Add requirements compliance details
        if requirements.get("summary"):
            comment += f"""### ✅ Requirements Compliance:
{requirements['summary']}

"""

        # Add security analysis details
        if security.get("summary"):
            comment += f"""### 🔒 Security & Best Practices:
{security['summary']}

"""

        # Add recommendation
        recommendation = overall.get('recommendation', 'needs_work')
        if recommendation == 'approve':
            comment += "### 🎉 Recommendation: **APPROVE** ✅\n"
        elif recommendation == 'approve_with_suggestions':
            comment += "### 💡 Recommendation: **APPROVE WITH SUGGESTIONS** ⚠️\n"
        elif recommendation == 'request_changes':
            comment += "### 🔄 Recommendation: **REQUEST CHANGES** 🔄\n"
        else:
            comment += "### ❌ Recommendation: **NEEDS SIGNIFICANT WORK** ❌\n"

        comment += "\n---\n*This review was generated automatically by AI Reviewer Agent*"
        
        return comment

    def _format_issues_comment(self, issues: List[Dict]) -> str:
        """Format issues into a separate comment."""
        if not issues:
            return ""
            
        comment = "## 🐛 Detailed Issues Found:\n\n"
        
        for issue in issues:
            severity = issue.get("type", "info").upper()
            emoji = "🔴" if severity == "ERROR" else "🟡" if severity == "WARNING" else "🔵"
            
            comment += f"{emoji} **{severity}**: {issue.get('message', 'No message')}\n"
            if issue.get("file"):
                comment += f"   📁 File: `{issue['file']}`"
                if issue.get("line"):
                    comment += f" (Line {issue['line']})"
                comment += "\n"
            comment += "\n"
        
        return comment

    def _extract_issue_number(self, title: str, body: str) -> Optional[int]:
        """Extract issue number from PR title or body."""
        # Look for patterns like "Fix #123", "Fixes #123", "Closes #123"
        patterns = [
            r'(?:fix|fixes|close|closes|resolve|resolves)\s*#(\d+)',
            r'#(\d+)',
        ]
        
        text = f"{title} {body}".lower()
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        return None