import json
from typing import Dict, Any

# Import your existing classifier logic
from predict import classify_driving_conditions


def classify_conditions(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps the Vehicle Safety Classifier logic in MCP-compatible format.
    """
    weather = input_data.get("weather")
    visibility = input_data.get("visibility")
    traffic = input_data.get("traffic")
    driver_state = input_data.get("driver_state")

    # Call your existing model (predict.py should have this logic)
    score, risk, explanation = classify_driving_conditions(
        weather, visibility, traffic, driver_state
    )

    return {
        "safety_score": score,
        "risk_level": risk,
        "explanation": explanation
    }


# Dispatcher to handle different MCP tool calls
def handle_mcp_request(tool_name: str, input_payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "classify_conditions":
        return classify_conditions(input_payload)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


if __name__ == "__main__":
    # Example standalone test
    test_payload = {
        "weather": "rain",
        "visibility": "low",
        "traffic": "heavy",
        "driver_state": "drowsy"
    }

    result = handle_mcp_request("classify_conditions", test_payload)
    print(json.dumps(result, indent=2))
