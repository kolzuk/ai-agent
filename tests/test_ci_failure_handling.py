"""Tests for strict CI failure handling functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from app.orchestrator import SDLCOrchestrator
from app.database import IssueIteration, IterationStatus


class TestCIFailureHandling:
    """Test cases for strict CI failure handling."""
    
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
    def mock_iteration_with_ci_failure(self):
        """Create a mock iteration with CI failure."""
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.repo_full_name = "owner/repo"
        iteration.issue_number = 123
        iteration.pr_number = 456
        iteration.current_iteration = 1
        iteration.max_iterations = 3
        iteration.installation_id = 789
        iteration.last_ci_status = "completed"
        iteration.last_ci_conclusion = "failure"
        iteration.pr_head_sha = "abc123"
        iteration.ci_status_source = "webhook"
        iteration.ci_status_updated_at = datetime.utcnow()
        return iteration
    
    @pytest.fixture
    def mock_review_result_approved(self):
        """Create a mock review result with approval."""
        return {
            "overall_assessment": {
                "score": 85,
                "recommendation": "approve",
                "summary": "Code looks good but CI failed"
            },
            "code_quality": {"summary": "Good code quality"},
            "requirements_compliance": {"summary": "Requirements met"},
            "security_analysis": {"summary": "No security issues"}
        }
    
    @pytest.mark.asyncio
    async def test_decide_next_action_ci_failure_with_approval(self, orchestrator, mock_iteration_with_ci_failure, 
                                                              mock_review_result_approved):
        """Test that approved code with CI failure starts new iteration."""
        with patch('app.orchestrator.db_manager') as mock_db, \
             patch.object(orchestrator, '_start_new_iteration_for_ci_failure') as mock_start_new:
            
            await orchestrator._decide_next_action(
                mock_iteration_with_ci_failure,
                mock_review_result_approved
            )
            
            # Should start new iteration for CI failure, not complete as successful
            mock_start_new.assert_called_once_with(
                mock_iteration_with_ci_failure,
                "failure",
                "webhook (recent)"
            )
    
    @pytest.mark.asyncio
    async def test_decide_next_action_ci_failure_max_iterations_reached(self, orchestrator, 
                                                                       mock_review_result_approved):
        """Test that CI failure at max iterations completes as failed."""
        # Create iteration at max iterations
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.current_iteration = 3
        iteration.max_iterations = 3
        iteration.last_ci_conclusion = "failure"
        iteration.pr_head_sha = "abc123"
        iteration.ci_status_source = "webhook"
        iteration.ci_status_updated_at = datetime.utcnow()
        
        with patch.object(orchestrator, '_complete_iteration') as mock_complete:
            await orchestrator._decide_next_action(iteration, mock_review_result_approved)
            
            # Should complete as failed, not successful
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args[0][1] == IterationStatus.FAILED  # Status should be FAILED
            assert "CI failed" in call_args[0][2]  # Message should mention CI failure
    
    @pytest.mark.asyncio
    async def test_decide_next_action_ci_success_with_approval(self, orchestrator, mock_review_result_approved):
        """Test that approved code with CI success completes successfully."""
        # Create iteration with CI success
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.current_iteration = 1
        iteration.max_iterations = 3
        iteration.last_ci_conclusion = "success"
        iteration.pr_head_sha = "abc123"
        iteration.ci_status_source = "webhook"
        iteration.ci_status_updated_at = datetime.utcnow()
        
        with patch.object(orchestrator, '_complete_iteration') as mock_complete:
            await orchestrator._decide_next_action(iteration, mock_review_result_approved)
            
            # Should complete as successful
            mock_complete.assert_called_once()
            call_args = mock_complete.call_args
            assert call_args[0][1] == IterationStatus.COMPLETED  # Status should be COMPLETED
            assert "approved" in call_args[0][2].lower()  # Message should mention approval
            assert "CI passed" in call_args[0][2]  # Message should mention CI success
    
    @pytest.mark.asyncio
    async def test_start_new_iteration_for_ci_failure(self, orchestrator, mock_iteration_with_ci_failure):
        """Test starting new iteration specifically for CI failure."""
        with patch('app.orchestrator.db_manager') as mock_db, \
             patch('app.orchestrator.get_github_client') as mock_get_client, \
             patch('asyncio.create_task') as mock_create_task:
            
            # Mock GitHub client
            mock_github = AsyncMock()
            mock_get_client.return_value.__aenter__.return_value = mock_github
            
            await orchestrator._start_new_iteration_for_ci_failure(
                mock_iteration_with_ci_failure,
                "failure",
                "webhook"
            )
            
            # Should update iteration with CI failure feedback
            mock_db.update_iteration.assert_called_once()
            update_call = mock_db.update_iteration.call_args
            assert "CI failed" in update_call[1]["last_review_feedback"]
            assert update_call[1]["status"] == IterationStatus.RUNNING
            
            # Should post comment to PR
            mock_github.create_issue_comment.assert_called_once()
            comment_call = mock_github.create_issue_comment.call_args
            assert "Starting New Iteration - CI Failed" in comment_call[0][2]
            
            # Should schedule next iteration
            mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_new_iteration_for_ci_failure_no_pr(self, orchestrator):
        """Test starting new iteration for CI failure when no PR exists."""
        # Create iteration without PR
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.pr_number = None
        iteration.current_iteration = 1
        
        with patch('app.orchestrator.db_manager') as mock_db, \
             patch('asyncio.create_task') as mock_create_task:
            
            await orchestrator._start_new_iteration_for_ci_failure(iteration, "failure", "webhook")
            
            # Should still update iteration and schedule next iteration
            mock_db.update_iteration.assert_called_once()
            mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_decide_next_action_uses_recent_webhook_data(self, orchestrator, mock_review_result_approved):
        """Test that recent webhook data is prioritized over API data."""
        # Create iteration with recent webhook data
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.repo_full_name = "owner/repo"
        iteration.pr_number = 456
        iteration.installation_id = 789
        iteration.current_iteration = 1
        iteration.max_iterations = 3
        iteration.pr_head_sha = "abc123"
        iteration.ci_status_source = "webhook"
        iteration.ci_status_updated_at = datetime.utcnow() - timedelta(minutes=2)  # Recent
        iteration.last_ci_conclusion = "failure"
        
        with patch.object(orchestrator, '_start_new_iteration_for_ci_failure') as mock_start_new:
            await orchestrator._decide_next_action(iteration, mock_review_result_approved)
            
            # Should use webhook data and start new iteration
            mock_start_new.assert_called_once_with(iteration, "failure", "webhook (recent)")
    
    @pytest.mark.asyncio
    async def test_decide_next_action_falls_back_to_api_data(self, orchestrator, mock_review_result_approved):
        """Test fallback to API data when webhook data is stale."""
        # Create iteration with stale webhook data
        iteration = MagicMock(spec=IssueIteration)
        iteration.id = 1
        iteration.repo_full_name = "owner/repo"
        iteration.pr_number = 456
        iteration.installation_id = 789
        iteration.current_iteration = 1
        iteration.max_iterations = 3
        iteration.pr_head_sha = "abc123"
        iteration.ci_status_source = "webhook"
        iteration.ci_status_updated_at = datetime.utcnow() - timedelta(minutes=10)  # Stale
        iteration.last_ci_conclusion = "failure"
        
        with patch('app.orchestrator.get_github_client') as mock_get_client, \
             patch('app.orchestrator.db_manager') as mock_db, \
             patch.object(orchestrator, '_start_new_iteration_for_ci_failure') as mock_start_new:
            
            # Mock GitHub client
            mock_github = AsyncMock()
            mock_get_client.return_value.__aenter__.return_value = mock_github
            mock_github.get_pull_request.return_value = {"head": {"sha": "abc123"}}
            mock_github.get_pr_ci_status.return_value = {
                "overall_conclusion": "failure",
                "overall_status": "completed"
            }
            
            await orchestrator._decide_next_action(iteration, mock_review_result_approved)
            
            # Should use API data and start new iteration
            mock_start_new.assert_called_once_with(iteration, "failure", "api (fallback)")
    
    @pytest.mark.asyncio
    async def test_decide_next_action_never_completes_successful_with_ci_failure(self, orchestrator):
        """Test that system never completes as successful when CI fails."""
        # Test various scenarios where CI fails
        ci_failure_scenarios = [
            ("failure", "webhook"),
            ("error", "api"),
            ("cancelled", "stored"),
            ("timed_out", "webhook")
        ]
        
        for ci_conclusion, data_source in ci_failure_scenarios:
            iteration = MagicMock(spec=IssueIteration)
            iteration.id = 1
            iteration.current_iteration = 1
            iteration.max_iterations = 3
            iteration.last_ci_conclusion = ci_conclusion
            iteration.pr_head_sha = "abc123"
            iteration.ci_status_source = data_source
            iteration.ci_status_updated_at = datetime.utcnow()
            
            review_result = {
                "overall_assessment": {
                    "recommendation": "approve",
                    "summary": "Code approved"
                }
            }
            
            with patch.object(orchestrator, '_complete_iteration') as mock_complete, \
                 patch.object(orchestrator, '_start_new_iteration_for_ci_failure') as mock_start_new:
                
                await orchestrator._decide_next_action(iteration, review_result)
                
                # Should never complete as successful with CI failure
                if mock_complete.called:
                    # If completed, should be FAILED, never COMPLETED
                    call_args = mock_complete.call_args
                    assert call_args[0][1] != IterationStatus.COMPLETED
                else:
                    # Should start new iteration instead
                    mock_start_new.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_ci_completion_with_failure(self, orchestrator):
        """Test handling CI completion event with failure status."""
        with patch('app.orchestrator.db_manager') as mock_db, \
             patch.object(orchestrator, '_run_review_iteration') as mock_run_review:
            
            # Mock iteration
            mock_iteration = MagicMock()
            mock_iteration.status = IterationStatus.WAITING_CI
            mock_db.get_iteration_by_pr.return_value = mock_iteration
            mock_db.get_iteration_by_pr.return_value = mock_iteration  # For verification
            
            result = await orchestrator.handle_ci_completion(
                repo_full_name="owner/repo",
                pr_number=456,
                ci_status="completed",
                ci_conclusion="failure",
                head_sha="abc123"
            )
            
            assert result is True
            
            # Should update CI status with failure
            mock_db.update_ci_status.assert_called_once()
            update_call = mock_db.update_ci_status.call_args
            assert update_call[1]["ci_conclusion"] == "failure"
            assert update_call[1]["pr_head_sha"] == "abc123"
            assert update_call[1]["source"] == "webhook"
            
            # Should start review process
            mock_run_review.assert_called_once()


@pytest.mark.integration
class TestCIFailureHandlingIntegration:
    """Integration tests for CI failure handling."""
    
    @pytest.mark.asyncio
    async def test_full_ci_failure_workflow(self):
        """Test the complete CI failure handling workflow."""
        # This would test the full flow from CI failure webhook
        # through review and decision making
        pass  # Placeholder for full integration test
    
    @pytest.mark.asyncio
    async def test_multiple_ci_failures_eventually_fail(self):
        """Test that multiple CI failures eventually result in failed iteration."""
        # This would test that after max iterations with CI failures,
        # the system properly fails the iteration
        pass  # Placeholder for comprehensive failure test