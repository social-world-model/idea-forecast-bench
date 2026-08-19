# Configuration for Idea Generation

# Keywords to search for in OpenReview
KEYWORDS = [
    "Diffusion Language Model",
    "Multimodal Visual Reasoning",
    "GUI /computer use / web agent",
    "Optimizer",
    "Time-series forecasting",
]

# OpenReview Settings
ICLR_YEAR = 2025
# Common Venue IDs:
# ICLR 2025: ICLR.cc/2025/Conference
# ICLR 2024: ICLR.cc/2024/Conference
VENUE_ID = f"ICLR.cc/{ICLR_YEAR}/Conference"

# Generation Settings
NUM_PAPERS_TO_FETCH = 5
MODEL = "gpt-4o-mini"  # Options: "gpt-4o", "gpt-4-turbo", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"
