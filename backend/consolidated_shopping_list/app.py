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

    meal_plan = body.get("meal_plan", "")

    prompt_text = (
        "You are a helpful meal prep assistant. Based on the meal plan below, "
        "generate a consolidated and organized shopping list.\n\n"
        f"Meal Plan: {meal_plan}\n\n"
        "Create a clean, practical shopping list that:\n"
        "- Groups ingredients by category (e.g. Produce, Proteins, Dairy, Grains, "
        "Pantry Staples, Spices and Condiments)\n"
        "- Combines duplicate ingredients and adjusts quantities where possible\n"
        "- Uses simple, store-friendly ingredient names\n"
        "- Avoids listing ingredients the user is allergic to or excluded by their "
        "dietary restrictions\n\n"
        "Make the list easy to follow on a phone while walking through a grocery store."
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
