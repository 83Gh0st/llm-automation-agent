#///script
# requires-python = ">=3.9"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "requests",
#   "pydantic",
#   "python-dotenv",
#   "python-dateutil"

# ]
#///

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import subprocess
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from datetime import datetime

load_dotenv()

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ✅ LLM Proxy URL
LLM_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# ✅ Define Tools for LLM Proxy
tools = [
    {
        "type": "function",
        "function": {
            "name": "script_runner",
            "description": "Install uv (if required) and run a script from a URL with provided arguments",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {"type": "string", "description": "The URL of the script to be executed"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of arguments to be passed to the script"
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_formatter",
            "description": "Format a markdown file using prettier@3.4.2",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the markdown file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_wednesdays",
            "description": "Count the number of Wednesdays in a given file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file containing dates"},
                    "output_file": {"type": "string", "description": "Path to save the Wednesday count"}
                },
                "required": ["file_path", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts.json by last_name, then first_name",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the contacts JSON file"},
                    "output_file": {"type": "string", "description": "Path to save the sorted JSON"}
                },
                "required": ["input_file", "output_file"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "extract_recent_logs",
        "description": "Extract the first line of the 10 most recent .log files from the logs directory",
        "parameters": {
            "type": "object",
            "properties": {
                "logs_directory": {
                    "type": "string",
                    "description": "Path to the directory containing .log files"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the extracted first lines of the logs"
                }
            },
            "required": ["logs_directory", "output_file"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "generate_markdown_index",
        "description": "Generate an index of Markdown files mapping filenames to their first H1 heading",
        "parameters": {
            "type": "object",
            "properties": {
                "docs_directory": {
                    "type": "string",
                    "description": "Path to the directory containing Markdown (.md) files"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the generated index in JSON format"
                }
            },
            "required": ["docs_directory", "output_file"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "extract_email_sender",
        "description": "Extract the sender's email address from an email message using LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "email_file": {
                    "type": "string",
                    "description": "Path to the email text file"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the extracted sender's email address"
                }
            },
            "required": ["email_file", "output_file"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "extract_credit_card_number",
        "description": "Extract the credit card number from an image using LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "image_file": {
                    "type": "string",
                    "description": "Path to the image file containing a credit card number"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the extracted credit card number without spaces"
                }
            },
            "required": ["image_file", "output_file"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "find_most_similar_comments",
        "description": "Find the most similar pair of comments using embeddings",
        "parameters": {
            "type": "object",
            "properties": {
                "comments_file": {
                    "type": "string",
                    "description": "Path to the file containing comments"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the most similar pair of comments"
                }
            },
            "required": ["comments_file", "output_file"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "calculate_gold_ticket_sales",
        "description": "Calculate the total sales for the 'Gold' ticket type from an SQLite database",
        "parameters": {
            "type": "object",
            "properties": {
                "database_file": {
                    "type": "string",
                    "description": "Path to the SQLite database file"
                },
                "output_file": {
                    "type": "string",
                    "description": "Path to save the total sales amount for 'Gold' ticket type"
                }
            },
            "required": ["database_file", "output_file"]
        }
    }
}


]

from dateutil import parser  # New import for improved date parsing

# ✅ Date Parsing Function (Supports Multiple Formats)
def parse_date(date_str):
    try:
        return parser.parse(date_str.strip())  # Uses dateutil to handle multiple formats
    except ValueError:
        raise ValueError(f"Unrecognized date format: {date_str}")
    

import cv2
import pytesseract
import re

def extract_credit_card_from_image(image_path: str) -> str:
    """
    Extracts a credit card number from an image using OCR.
    
    Parameters:
        image_path (str): Path to the image file containing the credit card.
    
    Returns:
        str: The extracted credit card number as a continuous string (no spaces).
    """
    # Load image in grayscale for better OCR performance
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not open image: {image_path}")

    # Preprocess the image (optional, improves accuracy)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # Improve contrast

    # Perform OCR
    extracted_text = pytesseract.image_to_string(image, config='--psm 6')

    # Use regex to find a valid credit card number (typically 16 digits)
    match = re.search(r"\b(?:\d[ -]*?){13,19}\b", extracted_text)
    
    if match:
        return re.sub(r"\D", "", match.group())  # Remove non-digit characters
    return ""

import sqlite3

def calculate_total_sales(db_path: str, output_file: str) -> None:
    """
    Calculates the total sales for the 'Gold' ticket type in an SQLite database.
    
    Parameters:
        db_path (str): Path to the SQLite database.
        output_file (str): Path to save the total sales amount.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to calculate total sales for 'Gold' tickets
        cursor.execute("""
            SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'
        """)
        total_sales = cursor.fetchone()[0] or 0  # Handle NULL cases

        # Write the result to output file
        with open(output_file, "w") as out_file:
            out_file.write(str(total_sales))

    except Exception as e:
        raise RuntimeError(f"Database query failed: {str(e)}")

    finally:
        conn.close()

from sentence_transformers import SentenceTransformer
import numpy as np

def find_most_similar_pair(file_path: str, output_file: str) -> None:
    """
    Finds the most similar pair of comments based on embeddings and writes them to a file.
    
    Parameters:
        file_path (str): Path to the file containing comments (one per line).
        output_file (str): Path to save the most similar pair of comments.
    """
    # Load comments
    with open(file_path, "r") as f:
        comments = [line.strip() for line in f if line.strip()]
    
    if len(comments) < 2:
        raise ValueError("Not enough comments to compare.")

    # Load a sentence embedding model (e.g., all-MiniLM)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments, convert_to_numpy=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings, embeddings.T)

    # Find the most similar pair (excluding self-comparison)
    np.fill_diagonal(similarity_matrix, -np.inf)
    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

    comment1, comment2 = comments[most_similar_indices[0]], comments[most_similar_indices[1]]

    # Write to output file
    with open(output_file, "w") as out_file:
        out_file.write(f"{comment1}\n{comment2}")


import re

def extract_email_from_text(text: str) -> str:
    """
    Extracts the sender's email address from the given email text.
    
    Parameters:
        text (str): The content of an email message.
    
    Returns:
        str: The extracted email address if found, otherwise an empty string.
    """
    email_pattern = re.compile(r"From:\s*.*?<?([\w\.-]+@[\w\.-]+\.\w+)>?", re.IGNORECASE)
    match = email_pattern.search(text)
    
    if match:
        return match.group(1)  # Extract the email address
    return ""  # Return empty string if no email found



# ✅ Task Execution Endpoint
@app.post("/run")
def task_runner(
    task: Optional[str] = Query(None, description="Task to execute"),  
    email: Optional[str] = Query(None, description="User email"),
    request_body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Runs a predefined task or delegates to LLM Proxy for interpretation.
    Supports both query parameters and JSON request body.
    """
    
    task_value = request_body.get("task") if request_body else task
    email_value = request_body.get("email") if request_body else email

    if not task_value:
        raise HTTPException(status_code=400, detail="Task is required")

    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
    if not AIPROXY_TOKEN:
        raise HTTPException(status_code=500, detail="AIPROXY_TOKEN is not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": task_value},
            {"role": "system", "content": "You are an assistant who executes tasks based on user instructions."}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }

    try:
        response = requests.post(LLM_PROXY_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        tool_calls = response_data.get("choices", [])[0].get("message", {}).get("tool_calls", [])
        if not tool_calls:
            raise HTTPException(status_code=500, detail="No tool calls found in LLM response")

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            if function_name == "script_runner":
                script_url = arguments["script_url"]
                email_arg = arguments["args"][0]

                try:
                    subprocess.run(["uv", "run", script_url, email_arg], check=True)
                except subprocess.CalledProcessError as e:
                    raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")

            elif function_name == "file_formatter":
                file_path = os.path.abspath(arguments["file_path"])

                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="File not found")

                try:
                    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True)
                except subprocess.CalledProcessError as e:
                    raise HTTPException(status_code=500, detail=f"Prettier formatting failed: {str(e)}")

            elif function_name == "count_wednesdays":
                file_path = os.path.abspath(arguments["file_path"])
                output_file = os.path.abspath(arguments["output_file"])

                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="File not found")

                try:
                    wednesday_count = 0
                    with open(file_path, "r") as file:
                        for line in file:
                            date_obj = parse_date(line.strip())
                            if date_obj.weekday() == 2:  # Wednesday
                                wednesday_count += 1

                    with open(output_file, "w") as out_file:
                        out_file.write(str(wednesday_count))

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing dates: {str(e)}")

            elif function_name == "sort_contacts":
                input_file = os.path.abspath(arguments["input_file"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    with open(input_file, "r") as file:
                        contacts = json.load(file)

                    sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))

                    with open(output_file, "w") as out_file:
                        json.dump(sorted_contacts, out_file, indent=4)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error sorting contacts: {str(e)}")

            elif function_name == "extract_log_headlines":
                log_dir = os.path.abspath(arguments["log_directory"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    log_files = sorted(
                        (f for f in os.listdir(log_dir) if f.endswith(".log")),
                        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
                        reverse=True
                    )[:10]

                    headlines = []
                    for log_file in log_files:
                        with open(os.path.join(log_dir, log_file), "r") as file:
                            first_line = file.readline().strip()
                            if first_line:
                                headlines.append(first_line)

                    with open(output_file, "w") as out_file:
                        out_file.write("\n".join(headlines))

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing logs: {str(e)}")

            elif function_name == "generate_markdown_index":
                docs_dir = os.path.abspath(arguments["docs_directory"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    index = {}
                    for md_file in os.listdir(docs_dir):
                        if md_file.endswith(".md"):
                            with open(os.path.join(docs_dir, md_file), "r") as file:
                                for line in file:
                                    if line.startswith("# "):
                                        index[md_file] = line[2:].strip()
                                        break

                    with open(output_file, "w") as out_file:
                        json.dump(index, out_file, indent=4)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing markdown files: {str(e)}")

            elif function_name == "extract_email_sender":
                email_file = os.path.abspath(arguments["email_file"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    with open(email_file, "r") as file:
                        content = file.read()

                    sender_email = extract_email_from_text(content)

                    with open(output_file, "w") as out_file:
                        out_file.write(sender_email)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error extracting email sender: {str(e)}")

            elif function_name == "extract_credit_card_number":
                image_file = os.path.abspath(arguments["image_file"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    card_number = extract_credit_card_from_image(image_file)

                    with open(output_file, "w") as out_file:
                        out_file.write(card_number)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error extracting credit card number: {str(e)}")

            elif function_name == "find_most_similar_comments":
                comments_file = os.path.abspath(arguments["comments_file"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    with open(comments_file, "r") as file:
                        comments = [line.strip() for line in file]

                    similar_pair = find_most_similar_pair(comments)

                    with open(output_file, "w") as out_file:
                        out_file.write("\n".join(similar_pair))

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error finding similar comments: {str(e)}")

            elif function_name == "calculate_gold_ticket_sales":
                database_file = os.path.abspath(arguments["database_file"])
                output_file = os.path.abspath(arguments["output_file"])

                try:
                    total_sales = calculate_total_sales(database_file, "Gold")

                    with open(output_file, "w") as out_file:
                        out_file.write(str(total_sales))

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error calculating gold ticket sales: {str(e)}")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"LLM Task Execution Failed: {str(e)}")

    return response_data
# ✅ API to Read Files
@app.get("/read")
def read_file(path: str = Query(..., description="Path of the file to read")):
    """
    Reads the contents of a file with error handling.
    """
    try:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found")

        with open(path, "r") as file:
            content = file.read()

        return {"status": "success", "content": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ✅ Run API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
