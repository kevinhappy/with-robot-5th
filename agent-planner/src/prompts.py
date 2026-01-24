from typing import Callable, List, Literal

from pydantic import BaseModel, Field


def make_goal_decomp_node_inputs(state):
    return {
        "object_text": state["inputs"]["object_text"],
        "user_query": state["user_queries"][-1],
    }


GOAL_DECOMP_NODE_PROMPT = """
# Instruction
You are the Goal-Level Planner in the MLDT pipeline.  
Your job is to decompose the user's command into independent high-level subgoals.

Definition of Terms:
- High-level goal: A distinct objective expressed without describing detailed actions.  
- Attribute-based decomposition: Splitting goals based on shared attributes of objects such as color, size, or shape.  
- Semantic grouping: Grouping by meaning or shared properties rather than by grammatical structure.

General Rules:
- Each subgoal must represent one independent, meaningful objective.
- If the user input contains multiple intentions, split them by meaning.
- Do not describe movement, manipulation steps, or low-level actions. These will be handled later.
- Keep each subgoal short, natural, and faithful to the original meaning.
- Preserve the user query's order.

Attribute-Based Rules:
- If the user's command involves categorizing, sorting, grouping, matching, or organizing objects based on attributes, then you must apply attribute-based decomposition.
- Extract object attributes from object_text. For example: "object_red_0" has the attribute "red".  
- Detect attribute groups (such as colors) from the object_text and match objects to bowls with the same attribute.
- When attribute-based organization is required, the number of subgoals must match the number of attribute groups.

Examples:
User input:
Organize the objects to the bowls according to their colors

Given object_text:
{{
    "object_name": "object_red_0",
    "object_name": "object_yellow_0",
    "object_name": "object_yellow_1",
    "object_name": "object_red_bowl_0",
    "object_name": "object_yellow_bowl_7",
}}

Output:
[
    "Organize the red objects to the red bowls",
    "Organize the yellow objects to the yellow bowls"
]

# Input
<object_text>
{object_text}
</object_text>
<user_query>
{user_query}
</user_query>

# Output Format
Return only the structured output following the JSON schema.
{format_instructions}
"""


class GoalDecompNodeParser(BaseModel):
    subgoals: List[str] = Field(
        ...,
        description="A list of high-level subgoals decomposed from the user query.",
    )


def make_task_decomp_node_inputs(
    state,
):
    def make_subgoals_text(subgoals):
        subgoals = subgoals.get("subgoals", [])

        return (
            "[\n"
            + "\n".join([f"{subgoal}" for i, subgoal in enumerate(subgoals)])
            + "\n]"
        )

    inputs = state.get("inputs", {})
    subgoals_text = make_subgoals_text(state.get("subgoals", []))
    print(f"Subgoals Text:\n{subgoals_text}\n")

    return {
        "skill_text": inputs.get("skill_text", ""),
        "object_text": inputs.get("object_text", ""),
        "subgoals_text": subgoals_text,
    }


TASK_DECOMP_NODE_PROMPT = """
# Role
You are the Task-Level Planner in the MLDT pipeline.
Your job is to convert a single high-level subgoal into an ordered sequence of semantic tasks that the robot can perform using its built-in skills.

Definition of Terms:
- Semantic task: A meaningful, minimal operation that contributes directly toward completing a subgoal.  
- Skill: A predefined robot capability such as moving to an object, picking an object, or placing an object.  
- Target: The object or location to which a skill is applied.

# Task-Level Principles
1. You must interpret the subgoal as a high-level objective that is already attribute-grouped by the Goal-Level Planner.
2. Your output must be a sequence of semantic tasks that use the robot's built-in skills.
3. Do not add new interpretations beyond the subgoal.  
4. Do not infer colors, groups, or attributes beyond what is explicitly present in the subgoal or object_text.
5. Do not describe low-level motion details. You must only specify which skill is used and which object is targeted.

# Required Behavior
- Use only the skills listed in <skill_text>.  
- Select objects only from <object_text>.  
- You may ignore objects that are not relevant to the subgoal.  
- The task steps must be short, natural, logically ordered, and directly connected to the subgoal.
- Each task step must include:
  - the skill name,
  - the target object or target location.

# Process
1. Analyze the subgoal.  
2. Identify relevant objects in <object_text> that appear in or logically correspond to the subgoal.  
3. Convert the subgoal into a sequential list of semantic tasks using robot skills.  
4. Ensure that the sequence achieves the subgoal without unnecessary steps.  

# Few-Shot Example
Input:
<skill_text>
["from robot1.skills import GoToObject, PickObject, PlaceObject"]
</skill_text>

<object_text>
{{
    "object_name": "object_red_0",
    "object_name": "object_yellow_0",
    "object_name": "object_yellow_1",
    "object_name": "object_red_bowl_0",
    "object_name": "object_yellow_bowl_7",
}}
</object_text>

<subgoals_text>
[
    "Organize the red objects to the red bowls",
    "Organize the yellow objects to the yellow bowls"
]
</subgoals_text>

Output:
[
    {{
        "task": "Organize the red objects to the red bowls",
        "subtasks": [
            {{"skill": "GoToObject", "target": "object_red_0"}},
            {{"skill": "PickObject", "target": "object_red_0"}},
            {{"skill": "GoToObject", "target": "object_red_bowl_0"}},
            {{"skill": "PlaceObject", "target": "object_red_bowl_0"}}
        ]
    }},
    {{
        "task": "Organize the yellow objects to the yellow bowls",
        "subtasks": [
            {{"skill": "GoToObject", "target": "object_yellow_0"}},
            {{"skill": "PickObject", "target": "object_yellow_0"}},
            {{"skill": "GoToObject", "target": "object_yellow_bowl_7"}},
            {{"skill": "PlaceObject", "target": "object_yellow_bowl_7"}},
            {{"skill": "GoToObject", "target": "object_yellow_1"}},
            {{"skill": "PickObject", "target": "object_yellow_1"}},
            {{"skill": "GoToObject", "target": "object_yellow_bowl_7"}},
            {{"skill": "PlaceObject", "target": "object_yellow_bowl_7"}}
            
        ]
    }}
]

# Input Components
1. robot_skills  
<skill_text>
{skill_text}
</skill_text>

2. observation  
<object_text>
{object_text}
</object_text>

3. subgoals  
<subgoals_text>
{subgoals_text}
</subgoals_text>

# Output Format
Return only the structured output matching the JSON schema.
{format_instructions}
"""


class SubTask(BaseModel):
    skill: Literal["GoToObject", "PickObject", "PlaceObject"] = Field(
        ..., description="The robot skill to be used for this task."
    )
    target: str = Field(
        ...,
        description="The target object",
    )


class SubGoal(BaseModel):
    subgoal: str
    tasks: List[SubTask] = Field(
        ...,
        description="An ordered list of semantic tasks to achieve the subgoal.",
    )


class TaskDecompNodeParser(BaseModel):
    task_outputs: List[SubGoal] = Field(
        ...,
        description="A list of subgoals each decomposed into semantic tasks.",
    )
