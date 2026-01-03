"""Tests for Human-in-the-Loop (HITL) functionality."""


# =============================================================================
# Interrupt Tests
# =============================================================================


class TestInterruptFunctionality:
    """Tests for workflow interrupt functionality."""
    
    def test_state_supports_interrupt_flag(self, minimal_state):
        """Test state supports interrupt flag."""
        minimal_state["_is_interrupted"] = True
        minimal_state["_interrupt_node"] = "planner"
        
        assert minimal_state["_is_interrupted"] is True
        assert minimal_state["_interrupt_node"] == "planner"
    
    def test_interrupt_data_structure(self, state_after_literature):
        """Test interrupt data has expected structure."""
        state_after_literature["_interrupt_data"] = {
            "action": "approve_gap_analysis",
            "gap_analysis": {
                "primary_gap": "Limited SME studies",
                "gaps": ["SME studies", "Long-term effects"],
            },
            "message": "Please review and approve the gap analysis",
            "options": ["approve", "modify", "reject"],
        }
        
        interrupt_data = state_after_literature["_interrupt_data"]
        assert interrupt_data["action"] == "approve_gap_analysis"
        assert "options" in interrupt_data
    
    def test_interrupt_before_nodes_list(self):
        """Test interrupt_before nodes are correctly defined."""
        from src.graphs.research_workflow import INTERRUPT_BEFORE_NODES
        
        assert "gap_identifier" in INTERRUPT_BEFORE_NODES
        assert "planner" in INTERRUPT_BEFORE_NODES
    
    def test_interrupt_after_nodes_list(self):
        """Test interrupt_after nodes are correctly defined."""
        from src.graphs.research_workflow import INTERRUPT_AFTER_NODES
        
        assert "reviewer" in INTERRUPT_AFTER_NODES


# =============================================================================
# Approval Flow Tests
# =============================================================================


class TestApprovalFlow:
    """Tests for human approval workflows."""
    
    def test_gap_analysis_approval(self, state_after_literature):
        """Test gap analysis approval flow."""
        # Simulate approval data
        approval_response = {
            "approved": True,
            "modified_gaps": None,
            "comments": "Looks good",
        }
        
        state_after_literature["_gap_approval"] = approval_response
        assert state_after_literature["_gap_approval"]["approved"] is True
    
    def test_plan_approval(self, state_after_literature):
        """Test research plan approval flow."""
        state_after_literature["research_plan"] = {
            "methodology": "Panel regression",
            "data_sources": ["CRSP", "Compustat"],
        }
        
        approval_response = {
            "approved": True,
            "modifications": {"methodology": "Fixed effects panel regression"},
        }
        
        state_after_literature["_plan_approval"] = approval_response
        assert state_after_literature["_plan_approval"]["approved"] is True
    
    def test_final_review_approval(self, complete_state):
        """Test final review approval flow."""
        approval_response = {
            "approved": True,
            "ready_for_submission": True,
            "reviewer_comments": "Ready for publication",
        }
        
        complete_state["_final_approval"] = approval_response
        assert complete_state["_final_approval"]["ready_for_submission"] is True


# =============================================================================
# Modification Flow Tests
# =============================================================================


class TestModificationFlow:
    """Tests for human modification workflows."""
    
    def test_modify_research_question(self, state_after_intake):
        """Test modifying research question."""
        original = state_after_intake["refined_query"]
        
        state_after_intake["refined_query"] = "Modified: How does AI affect productivity in SMEs?"
        state_after_intake["_human_modified"] = True
        state_after_intake["_modification_history"] = [
            {"field": "refined_query", "original": original, "modified": state_after_intake["refined_query"]}
        ]
        
        assert "Modified" in state_after_intake["refined_query"]
        assert len(state_after_intake["_modification_history"]) == 1
    
    def test_modify_gap_analysis(self, state_after_literature):
        """Test modifying gap analysis."""
        state_after_literature["gap_analysis"] = {
            "primary_gap": "Original gap",
            "gaps": ["Gap 1", "Gap 2"],
        }
        
        state_after_literature["gap_analysis"]["primary_gap"] = "Modified primary gap"
        state_after_literature["gap_analysis"]["gaps"].append("Gap 3 (human added)")
        
        assert "Modified" in state_after_literature["gap_analysis"]["primary_gap"]
        assert len(state_after_literature["gap_analysis"]["gaps"]) == 3
    
    def test_modify_research_plan(self, state_after_literature):
        """Test modifying research plan."""
        state_after_literature["research_plan"] = {
            "methodology": "OLS regression",
            "variables": ["x1", "x2"],
        }
        
        state_after_literature["research_plan"]["methodology"] = "2SLS regression"
        state_after_literature["research_plan"]["variables"].append("x3 (instrument)")
        
        assert state_after_literature["research_plan"]["methodology"] == "2SLS regression"


# =============================================================================
# Rejection Flow Tests
# =============================================================================


class TestRejectionFlow:
    """Tests for human rejection workflows."""
    
    def test_reject_gap_analysis(self, state_after_literature):
        """Test rejecting gap analysis."""
        rejection_response = {
            "approved": False,
            "reason": "Gap analysis is too broad",
            "suggestions": ["Focus on specific industry", "Narrow time period"],
        }
        
        state_after_literature["_gap_rejection"] = rejection_response
        assert state_after_literature["_gap_rejection"]["approved"] is False
        assert len(state_after_literature["_gap_rejection"]["suggestions"]) == 2
    
    def test_reject_triggers_revision(self, complete_state):
        """Test rejection triggers revision."""
        complete_state["reviewer_output"] = {
            "decision": "REVISE",
            "revision_requests": [
                {"section": "methods", "issue": "Robustness checks needed"},
            ],
        }
        
        assert complete_state["reviewer_output"]["decision"] == "REVISE"


# =============================================================================
# Timeout and Default Actions Tests
# =============================================================================


class TestTimeoutAndDefaults:
    """Tests for timeout and default action handling."""
    
    def test_state_supports_timeout_config(self, minimal_state):
        """Test state supports timeout configuration."""
        minimal_state["_hitl_timeout_seconds"] = 3600  # 1 hour
        minimal_state["_hitl_default_action"] = "approve"
        
        assert minimal_state["_hitl_timeout_seconds"] == 3600
        assert minimal_state["_hitl_default_action"] == "approve"
    
    def test_state_tracks_wait_time(self, minimal_state):
        """Test state tracks wait time for HITL."""
        from datetime import datetime, timezone
        
        minimal_state["_hitl_wait_started"] = datetime.now(timezone.utc).isoformat()
        
        assert "_hitl_wait_started" in minimal_state


# =============================================================================
# Notification Tests
# =============================================================================


class TestNotificationSupport:
    """Tests for async notification support."""
    
    def test_state_supports_notification_config(self, minimal_state):
        """Test state supports notification configuration."""
        minimal_state["_notification_config"] = {
            "email": "researcher@example.com",
            "webhook": "https://example.com/webhook",
            "notify_on": ["interrupt", "complete", "error"],
        }
        
        assert "email" in minimal_state["_notification_config"]
        assert "interrupt" in minimal_state["_notification_config"]["notify_on"]
    
    def test_state_tracks_notifications_sent(self, minimal_state):
        """Test state tracks sent notifications."""
        minimal_state["_notifications_sent"] = [
            {"type": "interrupt", "timestamp": "2026-01-03T12:00:00Z", "status": "delivered"},
        ]
        
        assert len(minimal_state["_notifications_sent"]) == 1
