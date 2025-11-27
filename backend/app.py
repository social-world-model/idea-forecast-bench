from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import threading
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend.idea_generator import generate_ideas
from backend import config

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend development

import json
from pathlib import Path

# ... (imports)

# Define path to generated ideas file
GENERATED_IDEAS_FILE = os.path.join(project_root, 'backend', 'generated_ideas.json')

@app.route('/api/generate-ideas', methods=['GET', 'POST'])
def api_generate_ideas():
    # Check if file exists
    if os.path.exists(GENERATED_IDEAS_FILE):
        print(f"Loading existing ideas from {GENERATED_IDEAS_FILE}")
        try:
            with open(GENERATED_IDEAS_FILE, 'r') as f:
                ideas = json.load(f)
            return jsonify(ideas)
        except Exception as e:
            print(f"Error reading existing file: {e}")
            # Fall through to generation if read fails
    
    # If we are here, file doesn't exist or couldn't be read.
    # We need to generate ideas.
    
    # Determine keywords and n
    if request.method == 'POST':
        data = request.json or {}
        keywords = data.get('keywords')
        n = data.get('n', config.NUM_PAPERS_TO_FETCH)
    else:
        # GET request or automatic generation
        keywords = config.KEYWORDS
        n = config.NUM_PAPERS_TO_FETCH
    
    print(f"Generating new ideas. Keywords: {keywords}, n: {n}")
    
    try:
        ideas = generate_ideas(keywords, n)
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(GENERATED_IDEAS_FILE), exist_ok=True)
            with open(GENERATED_IDEAS_FILE, 'w') as f:
                json.dump(ideas, f, indent=2)
            print(f"Saved generated ideas to {GENERATED_IDEAS_FILE}")
        except Exception as e:
            print(f"Error saving to file: {e}")
            
        return jsonify(ideas)
    except Exception as e:
        print(f"Error generating ideas: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/research-ideas', methods=['GET'])
def api_research_ideas():
    """
    Endpoint for the main dashboard.
    Returns ideas in the format expected by the frontend (ResearchIdea interface).
    """
    # Reuse the logic to get ideas (load from file or generate)
    # We can call the internal function or the API handler. 
    # Calling the API handler logic internally is cleaner.
    
    ideas = []
    if os.path.exists(GENERATED_IDEAS_FILE):
        try:
            with open(GENERATED_IDEAS_FILE, 'r') as f:
                ideas = json.load(f)
        except Exception as e:
            print(f"Error reading file for dashboard: {e}")
    
    # If no ideas found, we could trigger generation, but for dashboard load 
    # it might be better to return empty or wait for the background thread.
    # The background thread should have started generation on startup.
    
    transformed_ideas = []
    for idea in ideas:
        # Transform to match frontend ResearchIdea interface
        transformed = {
            "id": idea.get("id", f"gen_{os.urandom(4).hex()}"),
            "title": idea.get("Title", "Untitled Idea"),
            "description": f"**Problem:** {idea.get('Problem', '')}\n\n**Approach:** {idea.get('Approach', '')}",
            "author": "AI Researcher",
            "institution": "Live Idea Bench",
            "tags": ["AI Generated", "ICLR 2025"],
            "upvotes": int(idea.get("Score", 0)) * 10 + int(idea.get("Interestingness", 0)),
            "impact_score": idea.get("Score"),
            "created_at": datetime.now().isoformat(), # Dynamic timestamp for now
            "updated_at": datetime.now().isoformat(),
            "url": idea.get("source_url"),
            "citations": 0
        }
        
        # Add some dynamic tags based on scores
        if idea.get("Novelty", 0) > 8:
            transformed["tags"].append("High Novelty")
        if idea.get("Feasibility", 0) > 8:
            transformed["tags"].append("High Feasibility")
            
        transformed_ideas.append(transformed)
        
    return jsonify(transformed_ideas)

def run_initial_generation():
    """Checks if ideas exist, and if not, generates them in the background."""
    if not os.path.exists(GENERATED_IDEAS_FILE):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No generated ideas found. Starting automatic generation with default keywords...")
        try:
            # Use defaults from config
            keywords = config.KEYWORDS
            n = config.NUM_PAPERS_TO_FETCH
            
            ideas = generate_ideas(keywords, n)
            
            # Save to file
            os.makedirs(os.path.dirname(GENERATED_IDEAS_FILE), exist_ok=True)
            with open(GENERATED_IDEAS_FILE, 'w') as f:
                json.dump(ideas, f, indent=2)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Automatic generation complete. Saved to {GENERATED_IDEAS_FILE}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error during automatic generation: {e}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found existing generated ideas at {GENERATED_IDEAS_FILE}")

if __name__ == '__main__':
    print("*" * 50)
    print("Starting Live Idea Bench Backend")
    print(f"Generated Ideas File: {GENERATED_IDEAS_FILE}")
    print(f"Model: {config.MODEL}")
    print("Waiting for requests at /api/generate-ideas...")
    print("*" * 50)
    
    # Start initial generation in a background thread
    threading.Thread(target=run_initial_generation, daemon=True).start()
    
    app.run(debug=True, port=5000)
