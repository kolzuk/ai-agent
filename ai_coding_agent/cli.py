"""Command Line Interface for AI Coding Agent."""

import asyncio
import logging
import sys
from typing import Optional

import click
from dotenv import load_dotenv

from .code_agent import CodeAgent
from .config import config
from .reviewer_agent import ReviewerAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format=config.log_format,   
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ai_coding_agent.log")
    ]
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """AI Coding Agent - Automated GitHub SDLC system."""
    pass


@main.command()
@click.argument("issue_number", type=int)
@click.option(
    "--max-iterations",
    default=None,
    type=int,
    help="Maximum number of iterations for fixing issues"
)
def process_issue(issue_number: int, max_iterations: Optional[int]) -> None:
    """Process a GitHub issue and create a pull request with the solution.
    
    ISSUE_NUMBER: The GitHub issue number to process
    """
    click.echo(f"🚀 Processing issue #{issue_number}...")
    
    try:
        # Override max iterations if provided
        if max_iterations:
            config.max_iterations = max_iterations
            
        # Initialize and run code agent
        code_agent = CodeAgent()
        pr_number = asyncio.run(code_agent.process_issue(issue_number))
        
        if pr_number:
            click.echo(f"✅ Successfully created pull request #{pr_number}")
            click.echo(f"🔗 View at: {config.github_repo_url}/pull/{pr_number}")
        else:
            click.echo("❌ Failed to process issue")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error processing issue: {e}")
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@main.command()
@click.argument("pr_number", type=int)
def review_pr(pr_number: int) -> None:
    """Review a pull request and provide feedback.
    
    PR_NUMBER: The pull request number to review
    """
    click.echo(f"🔍 Reviewing pull request #{pr_number}...")
    
    try:
        # Initialize and run reviewer agent
        reviewer_agent = ReviewerAgent()
        result = asyncio.run(reviewer_agent.review_pull_request(pr_number))
        
        if result.get("status") == "completed":
            overall = result.get("overall_assessment", {})
            click.echo(f"✅ Review completed!")
            click.echo(f"📊 Score: {overall.get('score', 0)}/100")
            click.echo(f"💡 Recommendation: {overall.get('recommendation', 'unknown')}")
            click.echo(f"🔗 View at: {config.github_repo_url}/pull/{pr_number}")
        else:
            click.echo(f"❌ Review failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error reviewing PR: {e}")
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@main.command()
@click.argument("issue_number", type=int)
@click.option(
    "--max-iterations",
    default=None,
    type=int,
    help="Maximum number of iterations for the full cycle"
)
def full_cycle(issue_number: int, max_iterations: Optional[int]) -> None:
    """Run the full SDLC cycle: process issue -> create PR -> review -> iterate if needed.
    
    ISSUE_NUMBER: The GitHub issue number to process
    """
    click.echo(f"🔄 Starting full SDLC cycle for issue #{issue_number}...")
    
    try:
        max_iter = max_iterations or config.max_iterations
        iteration = 0
        
        code_agent = CodeAgent()
        reviewer_agent = ReviewerAgent()
        
        while iteration < max_iter:
            iteration += 1
            click.echo(f"\n📍 Iteration {iteration}/{max_iter}")
            
            # Step 1: Process issue and create/update PR
            click.echo("🚀 Processing issue...")
            pr_number = asyncio.run(code_agent.process_issue(issue_number))
            
            if not pr_number:
                click.echo("❌ Failed to create pull request")
                sys.exit(1)
                
            click.echo(f"✅ Pull request #{pr_number} created/updated")
            
            # Step 2: Review the PR
            click.echo("🔍 Reviewing pull request...")
            review_result = asyncio.run(reviewer_agent.review_pull_request(pr_number))
            
            if review_result.get("status") != "completed":
                click.echo(f"❌ Review failed: {review_result.get('message')}")
                sys.exit(1)
            
            overall = review_result.get("overall_assessment", {})
            score = overall.get("score", 0)
            recommendation = overall.get("recommendation", "unknown")
            
            click.echo(f"📊 Review Score: {score}/100")
            click.echo(f"💡 Recommendation: {recommendation}")
            
            # Step 3: Check if we should continue
            if recommendation in ["approve", "approve_with_suggestions"]:
                click.echo("🎉 Pull request approved! SDLC cycle completed successfully.")
                click.echo(f"🔗 Final PR: {config.github_repo_url}/pull/{pr_number}")
                break
            elif recommendation == "request_changes":
                if iteration < max_iter:
                    click.echo("🔄 Changes requested. Starting next iteration...")
                    continue
                else:
                    click.echo("⚠️ Maximum iterations reached. Manual intervention may be needed.")
                    break
            else:
                click.echo("❌ Significant issues found. Manual intervention required.")
                break
        
        if iteration >= max_iter:
            click.echo(f"⚠️ Reached maximum iterations ({max_iter}). Process stopped.")
            
    except Exception as e:
        logger.error(f"Error in full cycle: {e}")
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@main.command()
def config_info() -> None:
    """Display current configuration information."""
    click.echo("🔧 AI Coding Agent Configuration:")
    click.echo(f"📁 Repository: {config.github_repo_owner}/{config.github_repo_name}")
    click.echo(f"🤖 LLM Model: {config.openai_model}")
    click.echo(f"🔄 Max Iterations: {config.max_iterations}")
    click.echo(f"📝 Log Level: {config.log_level}")
    
    if config.use_yandex_gpt:
        click.echo("🇷🇺 Using Yandex GPT")
    else:
        click.echo("🤖 Using OpenAI GPT")


@main.command()
def validate_config() -> None:
    """Validate the current configuration."""
    click.echo("🔍 Validating configuration...")
    
    errors = []
    warnings = []
    
    # Check required environment variables
    if not config.github_token:
        errors.append("GITHUB_TOKEN is not set")
    
    if not config.github_repo_owner:
        errors.append("GITHUB_REPO_OWNER is not set")
        
    if not config.github_repo_name:
        errors.append("GITHUB_REPO_NAME is not set")
    
    if not config.use_yandex_gpt and not config.openai_api_key:
        errors.append("OPENAI_API_KEY is not set and Yandex GPT is not configured")
    
    if config.use_yandex_gpt:
        if not config.yandex_gpt_api_key:
            errors.append("YANDEX_GPT_API_KEY is set but incomplete")
        if not config.yandex_gpt_folder_id:
            errors.append("YANDEX_GPT_FOLDER_ID is set but incomplete")
    
    # Check configuration values
    if config.max_iterations < 1 or config.max_iterations > 10:
        warnings.append(f"MAX_ITERATIONS ({config.max_iterations}) should be between 1 and 10")
    
    # Display results
    if errors:
        click.echo("❌ Configuration errors found:")
        for error in errors:
            click.echo(f"  • {error}")
    
    if warnings:
        click.echo("⚠️ Configuration warnings:")
        for warning in warnings:
            click.echo(f"  • {warning}")
    
    if not errors and not warnings:
        click.echo("✅ Configuration is valid!")
    elif not errors:
        click.echo("✅ Configuration is valid (with warnings)")
    else:
        click.echo("❌ Configuration has errors that need to be fixed")
        sys.exit(1)


if __name__ == "__main__":
    main()