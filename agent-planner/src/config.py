from pydantic import BaseModel, ConfigDict



class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str
    prompt_dir: str


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    prompt_cache_key: str | None = None


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_decomp_node: NodeConfig
    task_decomp_node: NodeConfig


class RobotSkillConfig(BaseModel):
    name: str
    skills: list[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: PathsConfig
    runner: RunnerConfig
    skills: list[RobotSkillConfig]

config = Config(
    paths=PathsConfig(
        output_dir="output/",
        prompt_dir="src/graph/prompts/",
    ),
    runner=RunnerConfig(
        goal_decomp_node=NodeConfig(
            model_name="gpt41mini",
            prompt_cache_key="goal_decomp_node",
        ),
        task_decomp_node=NodeConfig(
            model_name="gpt41mini",
            prompt_cache_key="task_decomp_node",
        ),
    ),
    skills=[
        RobotSkillConfig(
            name="robot1",
            skills=["GoToObject", "PickObject", "PlaceObject"],
        )
    ],
)
