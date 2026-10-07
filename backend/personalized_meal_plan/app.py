import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_CLIENT = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def generate():
    body = request.get_json(force=True, silent=True) or {}

    dietary   = body.get("dietary_preferences", "")
    cuisine   = body.get("cuisine_preferences", "")
    num_days  = body.get("number_of_days", 7)
    meal_types = body.get("meal_types", "All meals")

    prompt_text = (
        "You are a friendly and knowledgeable meal planning assistant. "
        "Create a detailed, practical, and delicious meal plan based on the following user inputs.\n\n"
        f"Dietary preferences and restrictions: {dietary}\n"
        f"Cuisine preferences: {cuisine}\n"
        f"Number of days to plan: {num_days}\n"
        f"Meal types to include: {meal_types}\n\n"
        "Generate a well-organized meal plan covering the specified number of days and meal types. "
        "For each meal, include:\n"
        "- The meal name\n"
        "- A brief and appetizing description (1-2 sentences)\n"
        "- A short list of key ingredients\n\n"
        "Format the plan day by day, clearly labeled (e.g. Day 1, Day 2...). "
        "Make the meals varied, balanced, and aligned with the dietary restrictions and cuisine preferences provided. "
        "If no preferences are given, use a healthy balanced diet as the default."
    )

    # Build content blocks — support optional file upload
    content_blocks = []
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "")

    if file_data:
        if file_mime.startswith("image/"):
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })
        else:
            # Treat as document block
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })

    content_blocks.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content_blocks}]

    bedrock_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": messages,
    })

    def stream_response():
        resp = BEDROCK_CLIENT.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=bedrock_body,
        )
        for event in resp["body"]:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk["bytes"].decode("utf-8"))
                if chunk_data.get("type") == "content_block_delta":
                    delta = chunk_data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    return Response(
        stream_with_context(stream_response()),
        content_type="text/plain; charset=utf-8",
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
