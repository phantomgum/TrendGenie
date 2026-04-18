import os
import json
from dotenv import load_dotenv

load_dotenv()
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup

app = FastAPI()

class IdeaRequest(BaseModel):
    goal: str

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

@app.post("/generate-idea")
def generate_idea(request: IdeaRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    prompt = f"""
    You are an expert software engineer and creative project mentor. Your job is to design impressive, creative, and genuinely useful project ideas that force the developer to deeply learn a specific technology by building something real with it.

You are given a technology or tool the user wants to learn :{request.goal}. You generate ONE project idea that meets all of these criteria:

- CREATIVE: Not a todo app, not a clone, not a tutorial project. Something that would make someone say "that's actually a cool idea"
- SCOPED: Completable in a weekend to two weeks solo. Not a startup, not a platform. One focused thing.
- FORCED LEARNING: The technology must be central to the project — you cannot build it without deeply using that technology. It cannot be an afterthought.
- REAL UTILITY: It solves an actual problem or does something genuinely interesting. Not just a demo.
- IMPRESSIVE: The kind of project that looks good in a GitHub portfolio and sparks conversation.

Return your response as JSON with exactly these fields:
{{
  "title": "Short punchy project name",
  "tagline": "One sentence that makes someone want to build it",
  "description": "2-3 sentences on what it does and why it's interesting",
  "why_this_works": "1-2 sentences on why this project specifically forces deep learning of the technology",
  "core_features": ["feature 1", "feature 2", "feature 3"],
  "tools_and_tech": ["the requested technology", "other tools needed"],
  "what_youll_learn": ["specific skill 1", "specific skill 2", "specific skill 3"],
  "first_step": "The very first concrete thing to build — specific enough to start today",
  "estimated_time": "Realistic time range to complete"
}}

Return only valid JSON. No markdown, no backticks, no explanation outside the JSON.
    """

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = { 
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        content_text = data.get("content", [])[0].get("text", "")
        
        # Extract json safely in case Claude uses markdown blocks despite instructions
        if "```json" in content_text:
            content_text = content_text.split("```json")[1].split("```")[0].strip()
        elif "```" in content_text:
            content_text = content_text.split("```")[1].split("```")[0].strip()
            
        idea_json = json.loads(content_text)
        return idea_json
    except requests.exceptions.RequestException as e:
        # Anthropic provides a message in response JSON on failure usually
        err_msg = str(e)
        if e.response is not None:
             err_msg = e.response.text
        raise HTTPException(status_code=500, detail=f"Error communicating with Anthropic API: {err_msg}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse response from AI. Please try again.")

@app.get("/discover")
def discover_ideas():
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # 1. Scrape GitHub Trending
    try:
        github_res = requests.get("https://github.com/trending")
        github_res.raise_for_status()
        soup = BeautifulSoup(github_res.text, "html.parser")
        repos_html = soup.select("article.Box-row")[:10]
        trending_repos = []
        for repo in repos_html:
            name_elem = repo.select_one("h2 a")
            name = name_elem.text.strip().replace(" ", "").replace("\n", "") if name_elem else "Unknown"
            desc_elem = repo.select_one("p")
            description = desc_elem.text.strip() if desc_elem else "No description"
            lang_elem = repo.select_one("[itemprop='programmingLanguage']")
            language = lang_elem.text.strip() if lang_elem else "Unknown"
            trending_repos.append({"name": name, "description": description, "language": language})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape GitHub trending: {str(e)}")

    # 2. Hacker News Top Stories
    try:
        hn_url = "https://hacker-news.firebaseio.com/v1/topstories.json"
        hn_res = requests.get(hn_url)
        hn_res.raise_for_status()
        top_ids = hn_res.json()[:10]
        
        tech_keywords = ['tech', 'software', 'dev', 'ai', 'code', 'data', 'web', 'app', 'api', 'cloud', 'linux', 'startup', 'security', 'programming', 'release', 'framework', 'vulnerability', 'server', 'database', 'python', 'javascript', 'rust', 'github', 'open source', 'llm', 'model', 'cybersecurity', 'engineering', 'machine learning', 'algorithm']
        
        hn_stories = []
        for story_id in top_ids:
            story_res = requests.get(f"https://hacker-news.firebaseio.com/v1/item/{story_id}.json")
            story_data = story_res.json()
            if story_data and 'title' in story_data:
                title = story_data['title']
                if any(kw in title.lower() for kw in tech_keywords):
                    hn_stories.append(title)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Hacker News: {str(e)}")

    # 3. Format Context
    context_string = "Trending GitHub Repos:\n"
    for repo in trending_repos:
        context_string += f"- {repo['name']} ({repo['language']}): {repo['description']}\n"
    context_string += "\nTop Hacker News Tech Stories:\n"
    for story in hn_stories:
        context_string += f"- {story}\n"

    # 4. Generate Ideas from Context
    prompt = f"""You are an expert software engineer and creative project mentor who stays 
on the cutting edge of technology.

You will be given a list of currently trending GitHub repositories and 
top Hacker News stories. Your job is to synthesize what's genuinely 
hot right now and generate 3 creative, impressive project ideas that 
a developer could build to immerse themselves in these emerging tools 
and technologies.

Each idea must:
- Be directly inspired by or built with something from the trending data provided
- Be creative and non-obvious — not just "build a wrapper around X"
- Be scoped for one developer to complete in a weekend to two weeks
- Force deep hands-on learning with a technology that is actually trending right now
- Be genuinely useful or interesting, not just a demo

Here is what is currently trending:

{context_string}

Return exactly 3 project ideas as a JSON array. Each idea must have these fields:
{{
  "title": "Short punchy project name",
  "tagline": "One sentence that makes someone want to build it",
  "inspired_by": "Which trending repo or story inspired this",
  "description": "2-3 sentences on what it does and why it's interesting",
  "why_this_works": "Why this forces real learning of something trending",
  "core_features": ["feature 1", "feature 2", "feature 3"],
  "tools_and_tech": ["primary trending tech", "other tools"],
  "what_youll_learn": ["skill 1", "skill 2", "skill 3"],
  "first_step": "The very first concrete thing to build today",
  "estimated_time": "Realistic time range"
}}

Return only a valid JSON array of 3 objects. No markdown, no backticks, 
no explanation outside the JSON.
    """

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = { 
        "model": "claude-sonnet-4-6",
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        content_text = data.get("content", [])[0].get("text", "")
        
        if "```json" in content_text:
            content_text = content_text.split("```json")[1].split("```")[0].strip()
        elif "```" in content_text:
            content_text = content_text.split("```")[1].split("```")[0].strip()
            
        idea_json = json.loads(content_text)
        return idea_json
    except requests.exceptions.RequestException as e:
        err_msg = str(e)
        if e.response is not None:
             err_msg = e.response.text
        raise HTTPException(status_code=500, detail=f"Error communicating with Anthropic API: {err_msg}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse response from AI. Please try again.")

@app.get("/")
def read_root():
    # Read index.html relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
