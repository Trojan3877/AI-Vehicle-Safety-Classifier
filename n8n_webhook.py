from flask import Flask, request, jsonify
from mcp_adapter import handle_mcp_request

app = Flask(__name__)

@app.route("/n8n/classify", methods=["POST"])
def classify_webhook():
    """
    n8n webhook endpoint:
    Accepts JSON payload from n8n and routes to MCP adapter.
    """
    try:
        payload = request.get_json(force=True)
        tool_name = payload.get("tool", "classify_conditions")
        input_data = payload.get("input", {})

        result = handle_mcp_request(tool_name, input_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask for local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
