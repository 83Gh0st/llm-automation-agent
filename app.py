from flask import Flask, request, jsonify
import subprocess
import json
import os
import sqlite3
import datetime
from openai import OpenAI
from PIL import Image
import pytesseract

# Initialize Flask app
app = Flask(__name__)

# OpenAI API Key (Set this as an environment variable)
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Initialize OpenAI client with AI Proxy
try:
    client = OpenAI(api_key=AIPROXY_TOKEN, base_url="https://aiproxy.sanand.workers.dev/openai/")
except Exception as e:
    print("Error initializing OpenAI client:", e)

# Task parser using OpenAI LLM
def parse_task(task):
    prompt = f"Parse the following task and extract key action, input files, and the required operation:\n\n{task}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a task execution assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Function to execute predefined tasks
def execute_task(parsed_task):
    try:
        parsed_data = json.loads(parsed_task)
        action = parsed_data.get("action")
        input_file = parsed_data.get("input_file")
        output_file = parsed_data.get("output_file")

        if action == "install_uv_and_run":
            subprocess.run(["pip", "install", "uv"], check=True)
            subprocess.run(["python", "datagen.py", input_file], check=True)
            return {"message": "Data generation completed."}, 200

        elif action == "format_markdown":
            subprocess.run(["npx", "prettier@3.4.2", "--write", input_file], check=True)
            return {"message": "Markdown formatted."}, 200

        elif action == "count_wednesdays":
            with open(input_file, "r") as f:
                dates = [line.strip() for line in f.readlines()]
            count = sum(1 for date in dates if datetime.datetime.strptime(date, "%Y-%m-%d").weekday() == 2)
            with open(output_file, "w") as f:
                f.write(str(count))
            return {"message": f"Counted {count} Wednesdays."}, 200

        elif action == "sort_contacts":
            with open(input_file, "r") as f:
                contacts = json.load(f)
            sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
            with open(output_file, "w") as f:
                json.dump(sorted_contacts, f, indent=4)
            return {"message": "Contacts sorted."}, 200

        elif action == "extract_email":
            with open(input_file, "r") as f:
                email_content = f.read()
            prompt = f"Extract the sender's email address from this email:\n\n{email_content}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            email_address = response.choices[0].message.content.strip()
            with open(output_file, "w") as f:
                f.write(email_address)
            return {"message": "Email extracted."}, 200

        elif action == "extract_credit_card":
            img = Image.open(input_file)
            card_number = pytesseract.image_to_string(img).replace(" ", "").strip()
            with open(output_file, "w") as f:
                f.write(card_number)
            return {"message": "Credit card number extracted."}, 200

        else:
            return {"error": "Task not recognized."}, 400

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/run", methods=["POST"])
def run_task():
    task = request.args.get("task", "")
    if not task:
        return jsonify({"error": "No task provided."}), 400
    parsed_task = parse_task(task)
    response, status = execute_task(parsed_task)
    return jsonify(response), status

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
