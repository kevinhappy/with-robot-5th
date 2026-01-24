"""Planner state definitions and helpers."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import requests
from typing_extensions import TypedDict

from .config import Config, RobotSkillConfig, config


def make_object_text(url):
    response = requests.get(f"{url}/env")
    all = response.json()
    objects = all["objects"]
    print(objects)
    total_object_text = "{{\n"
    for obj in objects:
        object_text = f'"object_name": "{obj}",\n'
        total_object_text += object_text

    total_object_text += "}}"

    return total_object_text


def make_skill_text(config_skills: list[RobotSkillConfig]) -> str:
    skill_text_list = []
    for robot_skill in config_skills:

        skill_text = f"from {robot_skill.name}.skills import "
        for skill in robot_skill.skills:
            skill_text += f"{skill}"
            if skill != robot_skill.skills[-1]:
                skill_text += ", "
        skill_text_list.append(skill_text)

    return "\n".join(skill_text_list)


class StateSchema(TypedDict, total=False):
    """State contract for the planner LangGraph workflow."""

    user_queries: List[str]
    inputs: Dict[str, Any]
    subgoals: List[str]
    tasks: List[Dict[str, Any]]


def _make_base_state() -> StateSchema:
    return {
        "user_queries": [],
        "inputs": {},
        "subgoals": [],
        "tasks": [],
    }


def _make_inputs(config: Config, url: str) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    print("Making inputs for state...")
    inputs["object_text"] = make_object_text(url)
    inputs["skill_text"] = make_skill_text(config.skills)
    print(f"url: {url}")
    return inputs


def make_state(
    *, user_query: str, config: Config = config, url: None | str = None
) -> StateSchema:
    """Create a fresh state with defaults."""

    base_state = _make_base_state()
    resolved_url = url or "http://127.0.0.1:8800"

    state = copy.deepcopy(base_state)
    state["user_queries"] = [user_query]
    state["inputs"] = _make_inputs(config, resolved_url)
    return state
