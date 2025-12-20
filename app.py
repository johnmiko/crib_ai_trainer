"""FastAPI wrapper to expose cribbage agents via HTTP.

Endpoints
- GET /health
- GET /opponents
- POST /simulate

This is an initial lightweight API that runs simulated games between the
baseline agent and a selected opponent. Interactive human-vs-agent play can be
layered on later by adding stateful game sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
from Cribbage import Cribbage
from PlayerRandom import PlayerRandom
from Myrmidon import Myrmidon


@dataclass
class Opponent:
    key: str
    label: str
    description: str
    factory: Callable[[int], object]


def _build_opponents() -> Dict[str, Opponent]:
    return {
        "random": Opponent(
            key="random",
            label="Random",
            description="Baseline agent that selects legal moves at random.",
            factory=lambda seat: PlayerRandom(seat, False),
        ),
        "myrmidon": Opponent(
            key="myrmidon",
            label="Myrmidon",
            description="Heuristic rollouts (shallow search).",
            factory=lambda seat: Myrmidon(seat, 5, False),
        ),
    }


OPPONENTS = _build_opponents()

app = FastAPI(title="Cribbage API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulateRequest(BaseModel):
    opponent: str = Field(default="random", description="Opponent key from /opponents")
    games: int = Field(default=1, ge=1, le=20, description="How many games to simulate")


class SimulatedGame(BaseModel):
    id: str
    winner: int
    margin: int


class SimulateResponse(BaseModel):
    opponent: str
    games: int
    wins: Dict[str, int]
    results: List[SimulatedGame]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/opponents")
def list_opponents() -> List[Dict[str, str]]:
    return [
        {
            "key": opp.key,
            "label": opp.label,
            "description": opp.description,
        }
        for opp in OPPONENTS.values()
    ]


def _build_opponent(key: str, seat: int):
    opponent = OPPONENTS.get(key)
    if not opponent:
        raise HTTPException(status_code=404, detail=f"Unknown opponent '{key}'")
    return opponent.factory(seat)


@app.post("/simulate", response_model=SimulateResponse)
def simulate_games(req: SimulateRequest) -> SimulateResponse:
    wins = {"player": 0, "opponent": 0}
    results: List[SimulatedGame] = []

    for _ in range(req.games):
        game_id = str(uuid4())
        player = PlayerRandom(1, False)
        opponent = _build_opponent(req.opponent, 2)
        game = Cribbage([player, opponent], verboseFlag=False)
        diff = game.playGame()
        winner = 1 if diff > 0 else 2
        if winner == 1:
            wins["player"] += 1
        else:
            wins["opponent"] += 1
        results.append(
            SimulatedGame(id=game_id, winner=winner, margin=abs(int(diff)))
        )

    return SimulateResponse(
        opponent=req.opponent, games=req.games, wins=wins, results=results
    )
