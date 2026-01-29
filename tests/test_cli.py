"""Tests for CLI module."""

import os
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from ai_coding_agent.cli import main, config_info, validate_config


class TestCLI:
    """Test CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_command_help(self):
        """Test main command help output."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'AI Coding Agent' in result.output
        assert 'Automated GitHub SDLC system' in result.output

    def test_config_info_command(self):
        """Test config info command."""
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }):
            result = self.runner.invoke(config_info)
            assert result.exit_code == 0
            assert 'AI Coding Agent Configuration' in result.output
            assert 'test_owner/test_repo' in result.output

    def test_validate_config_success(self):
        """Test successful config validation."""
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }):
            result = self.runner.invoke(validate_config)
            assert result.exit_code == 0
            assert 'Configuration is valid' in result.output

    def test_validate_config_missing_token(self):
        """Test config validation with missing GitHub token."""
        with patch.dict(os.environ, {
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }, clear=True):
            result = self.runner.invoke(validate_config)
            assert result.exit_code == 1
            assert 'GITHUB_TOKEN is not set' in result.output

    @patch('ai_coding_agent.cli.CodeAgent')
    @patch('ai_coding_agent.cli.asyncio.run')
    def test_process_issue_command(self, mock_asyncio_run, mock_code_agent):
        """Test process issue command."""
        # Mock the code agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.process_issue.return_value = 123
        mock_code_agent.return_value = mock_agent_instance
        mock_asyncio_run.return_value = 123

        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }):
            result = self.runner.invoke(main, ['process-issue', '42'])
            assert result.exit_code == 0
            assert 'Processing issue #42' in result.output
            assert 'Successfully created pull request #123' in result.output

    @patch('ai_coding_agent.cli.ReviewerAgent')
    @patch('ai_coding_agent.cli.asyncio.run')
    def test_review_pr_command(self, mock_asyncio_run, mock_reviewer_agent):
        """Test review PR command."""
        # Mock the reviewer agent
        mock_agent_instance = MagicMock()
        mock_review_result = {
            'status': 'completed',
            'overall_assessment': {
                'score': 85,
                'recommendation': 'approve'
            }
        }
        mock_agent_instance.review_pull_request.return_value = mock_review_result
        mock_reviewer_agent.return_value = mock_agent_instance
        mock_asyncio_run.return_value = mock_review_result

        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }):
            result = self.runner.invoke(main, ['review-pr', '123'])
            assert result.exit_code == 0
            assert 'Reviewing pull request #123' in result.output
            assert 'Review completed' in result.output
            assert 'Score: 85/100' in result.output

    @patch('ai_coding_agent.cli.CodeAgent')
    @patch('ai_coding_agent.cli.ReviewerAgent')
    @patch('ai_coding_agent.cli.asyncio.run')
    def test_full_cycle_command_success(self, mock_asyncio_run, mock_reviewer_agent, mock_code_agent):
        """Test successful full cycle command."""
        # Mock code agent
        mock_code_instance = MagicMock()
        mock_code_instance.process_issue.return_value = 123
        mock_code_agent.return_value = mock_code_instance

        # Mock reviewer agent
        mock_reviewer_instance = MagicMock()
        mock_review_result = {
            'status': 'completed',
            'overall_assessment': {
                'score': 90,
                'recommendation': 'approve'
            }
        }
        mock_reviewer_instance.review_pull_request.return_value = mock_review_result
        mock_reviewer_agent.return_value = mock_reviewer_instance

        # Mock asyncio.run to return appropriate values
        def mock_run(coro):
            if hasattr(coro, '__name__') and 'process_issue' in str(coro):
                return 123
            else:
                return mock_review_result

        mock_asyncio_run.side_effect = mock_run

        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO_OWNER': 'test_owner',
            'GITHUB_REPO_NAME': 'test_repo',
            'OPENAI_API_KEY': 'test_openai_key',
        }):
            result = self.runner.invoke(main, ['full-cycle', '42'])
            assert result.exit_code == 0
            assert 'Starting full SDLC cycle' in result.output
            assert 'Pull request approved' in result.output

    def test_command_with_invalid_issue_number(self):
        """Test command with invalid issue number."""
        result = self.runner.invoke(main, ['process-issue', 'invalid'])
        assert result.exit_code != 0

    def test_max_iterations_option(self):
        """Test max iterations option."""
        with patch('ai_coding_agent.cli.CodeAgent') as mock_code_agent, \
             patch('ai_coding_agent.cli.asyncio.run') as mock_asyncio_run:
            
            mock_agent_instance = MagicMock()
            mock_agent_instance.process_issue.return_value = 123
            mock_code_agent.return_value = mock_agent_instance
            mock_asyncio_run.return_value = 123

            with patch.dict(os.environ, {
                'GITHUB_TOKEN': 'test_token',
                'GITHUB_REPO_OWNER': 'test_owner',
                'GITHUB_REPO_NAME': 'test_repo',
                'OPENAI_API_KEY': 'test_openai_key',
            }):
                result = self.runner.invoke(main, ['process-issue', '42', '--max-iterations', '3'])
                assert result.exit_code == 0