from fastapi import FastAPI
from datetime import date

app = FastAPI(title="Mock NFL MCP Server")

MOCK_PASSING_LEADERS = {
    "season": 2025,
    "as_of": str(date.today()),
    "leader": {
        "player": "Evan Carter",
        "team": "Metro City Hawks",
        "yards": 4186,
        "touchdowns": 29,
        "interceptions": 11
    }
}

MOCK_RUSHING_LEADERS = {
    "season": 2025,
    "as_of": str(date.today()),
    "leader": {
        "player": "Darius Knox",
        "team": "Iron Valley Wolves",
        "yards": 1389,
        "touchdowns": 10
    }
}

@app.get("/passing-leaders")
def passing_leaders():
    return MOCK_PASSING_LEADERS

@app.get("/rushing-leaders")
def rushing_leaders():
    return MOCK_RUSHING_LEADERS

@app.get("/health")
def health():
    return {"status": "ok"}