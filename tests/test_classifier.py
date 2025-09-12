import pytest
from mcp_adapter import handle_mcp_request


def test_high_risk_classification():
    payload = {
        "weather": "rain",
        "visibility": "low",
        "traffic": "heavy",
        "driver_state": "drowsy"
    }
    result = handle_mcp_request("classify_conditions", payload)
    assert "safety_score" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["low", "medium", "high"]
    assert result["risk_level"] == "high"


def test_low_risk_classification():
    payload = {
        "weather": "clear",
        "visibility": "high",
        "traffic": "light",
        "driver_state": "alert"
    }
    result = handle_mcp_request("classify_conditions", payload)
    assert result["risk_level"] == "low"
    assert result["safety_score"] >= 80


def test_invalid_tool():
    payload = {
        "weather": "snow",
        "visibility": "medium",
        "traffic": "moderate",
        "driver_state": "alert"
    }
    result = handle_mcp_request("invalid_tool", payload)
    assert "error" in result
    assert "Unknown tool" in result["error"]
