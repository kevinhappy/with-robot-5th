# Robot Agent – Goal/Task Decomposition

Two-stage LangGraph agent that decomposes a user command into subgoals and executable tasks. FastAPI server provides HTTP endpoints for LLM-based command processing and task execution.

## Overview
- Two nodes: goal decomposition → task decomposition.
- Uses OpenAI-compatible LLMs (`langgraph`, `langchain`) with Pydantic parsing.
- Pulls environment objects from a local simulator API (`http://127.0.0.1:8800/env`) and formats available robot skills from config.
- TaskExecutor sends HTTP actions to the same simulator.
- FastAPI server at port 8900 provides `/llm_command` endpoint for processing user commands.

## Architecture
```
START
  ↓
goal_decomp    # Break user query into subgoals using object context
  ↓
task_decomp    # Turn each subgoal into skill-level task lists
  ↓
END
```

| Node | Purpose | Output key |
|------|---------|------------|
| `goal_decomp` | Attribute-aware subgoal splitter based on `object_text` and the latest `user_query`. | `subgoals` |
| `task_decomp` | Maps each subgoal to ordered tasks that only use skills defined in config. | `tasks` |

State fields (from `StateSchema`): `user_queries`, `inputs` (`object_text`, `skill_text`), `subgoals`, `tasks`.

## Project Structure
```
agent/
├── main.py              # FastAPI server with /llm_command endpoint
├── environment.yml      # Conda environment configuration
├── test.ipynb           # Testing notebook
├── ui.html              # Web UI for the API server
├── src/
│   ├── __init__.py
│   ├── config.py        # Pydantic config models (PathsConfig, RunnerConfig, RobotSkillConfig)
│   ├── enums.py         # ModelNames enum (gpt-4.1, gpt-5, etc.)
│   ├── executor.py      # TaskExecutor - executes tasks via HTTP API
│   ├── graph.py         # LangGraph workflow creation and LLM node builders
│   ├── prompts.py       # Goal/task decomposition prompt templates
│   ├── state.py         # StateSchema and state initialization helpers
│   └── utils.py         # File I/O utilities (load/save for json, yaml, pkl, etc.)
└── README.md
```

## Core Modules

### config.py
Pydantic-based configuration management:
- `PathsConfig`: output_dir, prompt_dir
- `NodeConfig`: model_name, prompt_cache_key
- `RunnerConfig`: goal_decomp_node, task_decomp_node
- `RobotSkillConfig`: name, skills list
- `Config`: Main configuration class with paths, runner, and skills

Default configuration includes:
- Model: gpt41mini (gpt-4.1-mini)
- Skills: GoToObject, PickObject, PlaceObject

### enums.py
Model name definitions:
- `ModelNames` enum: gpt41, gpt41mini, gpt5, gpt5mini, gpt5nano, gpt4omini, gpt4o

### state.py
State management for LangGraph workflow:
- `StateSchema`: TypedDict with user_queries, inputs, subgoals, tasks
- `make_state()`: Initialize state from config and user query
- `make_object_text()`: Fetch and format environment objects from simulator API
- `make_skill_text()`: Format available robot skills for prompts

### graph.py
LangGraph workflow construction:
- `create_llm()`: Initialize ChatOpenAI with model configuration and caching
- `make_llm_node()`: Create LLM processing nodes with prompt templates and parsers
- `create_graph()`: Wire goal_decomp and task_decomp nodes into StateGraph

### prompts.py
Prompt templates and output schemas:
- `GOAL_DECOMP_NODE_PROMPT`: Attribute-based goal decomposition instructions
- `TASK_DECOMP_NODE_PROMPT`: Skill-level task generation instructions
- Pydantic models for structured output parsing

### executor.py
Task execution against simulator:
- `TaskExecutor`: Connects to simulator API (default: http://127.0.0.1:8800)
- `_make_task_sequence()`: Process task outputs into ordered sequence
- `_go_to_object()`, `_pick_object()`, `_place_object()`: Send HTTP actions
- `execute()`: Execute full task sequence

### utils.py
File I/O utilities:
- `load()`: Load files (txt, csv, json, yaml, pkl)
- `save()`: Save files with automatic directory creation
- `safe_print()`: Pretty print utility with error handling

## Execution

### Option 1: FastAPI Server (Recommended)
Start the server which provides a web UI and REST API:

```bash
# Install dependencies
conda env create -f environment.yml
conda activate robot_agent

# Ensure simulator is running at http://127.0.0.1:8800
# Start the API server
python main.py
```

The server runs at `http://0.0.0.0:8900`:
- `/`: Web UI (ui.html)
- `/llm_command`: POST endpoint for command processing

Example request:
```bash
curl -X POST "http://localhost:8900/llm_command" \
  -H "Content-Type: application/json" \
  -d '{"command": "Organize the objects to the bowls according to their colors"}'
```

### Option 2: Programmatic Usage
Use the agent modules directly in your code:

```python
from agent.src.config import config
from agent.src.state import make_state
from agent.src.graph import create_graph
from agent.src.executor import TaskExecutor

# Initialize graph with default config
graph = create_graph(config)

# Create state with user query
state = make_state(
    config=config,
    user_query="Organize the objects to the bowls according to their colors",
    url="http://127.0.0.1:8800"
)

# Run the graph
final_state = graph.invoke(state)

# Access results
subgoals = final_state["subgoals"]
task_outputs = final_state["tasks"]

# Execute tasks (optional)
executor = TaskExecutor(url="http://127.0.0.1:8800")
executor.execute(task_outputs)
```

### Option 3: Notebook
Use `test.ipynb` for interactive experimentation and testing.

## Configuration

The default configuration in `config.py`:

```python
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
```

You can modify the configuration directly in code or extend it with additional skills and settings.

## Dependencies

| Library | Purpose |
|---------|---------|
| `langchain` / `langgraph` | LLM orchestration and workflow graph |
| `langchain-openai` | OpenAI chat models integration |
| `pydantic` | Config and output validation |
| `requests` | Environment/simulator HTTP calls |
| `fastapi` / `uvicorn` | Web API server |
| `loguru` | Logging |
| `elevenlabs` | Text-to-speech (optional) |

## Environment Variables

Create a `.env` file for optional features:
```
ELEVENLABS_API_KEY=your_api_key_here
```

## License
See `../LICENSE`.
