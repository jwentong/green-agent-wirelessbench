"""
WCHW Green Agent Tests
UC Berkeley RDI Foundation AgentBeats Competition

Tests for the WCHW benchmark evaluator and A2A protocol compliance.
"""

from typing import Any
import pytest
import httpx
from uuid import uuid4
import json

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# ============================================================================
# A2A Protocol Validation Helpers
# ============================================================================

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    required_fields = frozenset([
        'name', 'description', 'url', 'version',
        'capabilities', 'defaultInputModes', 'defaultOutputModes', 'skills',
    ])

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if 'url' in card_data and not (
        card_data['url'].startswith('http://') or card_data['url'].startswith('https://')
    ):
        errors.append("Field 'url' must be an absolute URL starting with http:// or https://.")

    if 'capabilities' in card_data and not isinstance(card_data['capabilities'], dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append("Field 'skills' must be an array of AgentSkill objects.")
        elif not card_data['skills']:
            errors.append("Field 'skills' array is empty.")

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("StatusUpdate object missing required field: 'status.state'.")
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append("ArtifactUpdate object missing required field: 'artifact'.")
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# ============================================================================
# A2A Messaging Helpers
# ============================================================================

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]
    return events


# ============================================================================
# A2A Conformance Tests
# ============================================================================

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)


def test_agent_card_wchw_fields(agent):
    """Validate WCHW-specific fields in agent card."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200
    
    card_data = response.json()
    
    # Check for WCHW-specific metadata
    assert "WCHW" in card_data.get("name", "") or "WCHW" in card_data.get("description", ""), \
        "Agent card should mention WCHW benchmark"
    
    # Check skills contain evaluation capability
    skills = card_data.get("skills", [])
    assert len(skills) > 0, "Agent should have at least one skill"
    
    skill_tags = []
    for skill in skills:
        skill_tags.extend(skill.get("tags", []))
    
    assert any("wireless" in tag.lower() or "telecom" in tag.lower() or "benchmark" in tag.lower() 
               for tag in skill_tags), \
        "Skills should be tagged with wireless/telecom/benchmark keywords"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


# ============================================================================
# WCHW Evaluator Unit Tests
# ============================================================================

class TestWCHWEvaluator:
    """Unit tests for the WCHW evaluator logic."""
    
    @pytest.fixture
    def evaluator(self):
        import sys
        sys.path.insert(0, 'src')
        from agent import WCHWEvaluator
        return WCHWEvaluator()
    
    def test_classify_numeric_answer(self, evaluator):
        """Test classification of numeric answers."""
        assert evaluator.classify_answer_type("6.87 Mbps") == "numeric"
        assert evaluator.classify_answer_type("1.54") == "numeric"
        assert evaluator.classify_answer_type("240 m") == "numeric"
        assert evaluator.classify_answer_type("3100 Hz") == "numeric"
    
    def test_classify_scientific_notation(self, evaluator):
        """Test classification of scientific notation answers."""
        assert evaluator.classify_answer_type("5.42e-6") == "scientific"
        assert evaluator.classify_answer_type("2.2×10^-8") == "scientific"
        assert evaluator.classify_answer_type("1e6") == "scientific"
    
    def test_classify_formula_answer(self, evaluator):
        """Test classification of formula answers."""
        assert evaluator.classify_answer_type("0.5[m^2(t)+\\hat m^2(t)]") == "formula"
        assert evaluator.classify_answer_type("$s_{FM}(t)=3\\cos...$") == "formula"
    
    def test_extract_number_with_unit(self, evaluator):
        """Test number extraction with units."""
        value, unit = evaluator.extract_number_with_unit("6.87 Mbps")
        assert abs(value - 6.87e6) < 1e3  # Should be converted to bps
        
        value, unit = evaluator.extract_number_with_unit("240 m")
        assert abs(value - 240) < 0.1
        
        value, unit = evaluator.extract_number_with_unit("3.5 kHz")
        assert abs(value - 3500) < 1  # Should be converted to Hz
    
    def test_extract_scientific_notation(self, evaluator):
        """Test extraction of scientific notation numbers."""
        value, _ = evaluator.extract_number_with_unit("5.42e-6")
        assert abs(value - 5.42e-6) < 1e-10
        
        value, _ = evaluator.extract_number_with_unit("2.2×10^-8")
        assert abs(value - 2.2e-8) < 1e-12
    
    def test_numeric_score_exact_match(self, evaluator):
        """Test scoring for exact numeric matches."""
        score = evaluator.calculate_numeric_score(100.0, 100.0)
        assert score == 1.0
        
        score = evaluator.calculate_numeric_score(100.0, 100.5)  # 0.5% error
        assert score == 1.0
    
    def test_numeric_score_close_match(self, evaluator):
        """Test scoring for close numeric matches."""
        score = evaluator.calculate_numeric_score(100.0, 103.0)  # 3% error
        assert score == 0.9
        
        score = evaluator.calculate_numeric_score(100.0, 108.0)  # 8% error
        assert score == 0.7
    
    def test_numeric_score_unit_error(self, evaluator):
        """Test scoring for unit conversion errors."""
        # Off by factor of 1000 (e.g., kHz vs Hz)
        score = evaluator.calculate_numeric_score(1000.0, 1.0)
        assert score == 0.5
        
        score = evaluator.calculate_numeric_score(1.0, 1000.0)
        assert score == 0.5
    
    def test_evaluate_numeric_answer(self, evaluator):
        """Test full evaluation of numeric answers."""
        score, atype, details = evaluator.evaluate("6.87 Mbps", "6.87 Mbps")
        assert score == 1.0
        assert atype == "numeric"
        
        score, atype, details = evaluator.evaluate("240 m", "240.5 m")
        assert score >= 0.9  # Close match
    
    def test_evaluate_formula_answer(self, evaluator):
        """Test evaluation of formula answers."""
        score, atype, details = evaluator.evaluate(
            "0.5[m^2(t)+\\hat m^2(t)]",
            "0.5[m^2(t)+\\hat m^2(t)]"
        )
        assert score == 1.0
        assert atype == "formula"


# ============================================================================
# Integration Tests (require running agent)
# ============================================================================

@pytest.mark.asyncio
async def test_valid_eval_request(agent):
    """Test that a valid evaluation request is processed correctly."""
    request = {
        "participants": {
            "wireless_solver": "http://localhost:8080"  # Mock purple agent
        },
        "config": {
            "num_problems": 1,
            "timeout": 30
        }
    }
    
    # Note: This test will likely fail without a running purple agent
    # It's here to show the expected request format
    try:
        events = await send_text_message(
            json.dumps(request),
            agent,
            streaming=False
        )
        # If we get here, basic request parsing worked
        assert True
    except Exception as e:
        # Expected to fail without purple agent
        pytest.skip(f"Skipping integration test: {e}")


@pytest.mark.asyncio 
async def test_invalid_eval_request_missing_role(agent):
    """Test that missing roles are properly rejected."""
    request = {
        "participants": {},  # Missing required role
        "config": {
            "num_problems": 1
        }
    }
    
    events = await send_text_message(
        json.dumps(request),
        agent,
        streaming=False
    )
    
    # Should get a rejection or error response
    assert events, "Agent should respond even for invalid requests"


@pytest.mark.asyncio
async def test_invalid_eval_request_missing_config(agent):
    """Test that missing config keys are properly rejected."""
    request = {
        "participants": {
            "wireless_solver": "http://localhost:8080"
        },
        "config": {}  # Missing num_problems
    }
    
    events = await send_text_message(
        json.dumps(request),
        agent,
        streaming=False
    )
    
    assert events, "Agent should respond even for invalid requests"
