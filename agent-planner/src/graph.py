"""Planner graph wiring the LangGraph Goal -> Task pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from loguru import logger

from . import prompts as prompt_module
from .enums import ModelNames
from .state import StateSchema

StateCallable = Callable[[Any], Any]


def create_llm(
    model_name: ModelNames,
    temperature: float | None = None,
    prompt_cache_key: str | None = None,
):
    extra_body: Dict[str, Any] | None = None
    if prompt_cache_key:
        extra_body = {"prompt_cache_key": prompt_cache_key}
    llm_kwargs: Dict[str, Any] = {"model": model_name.value}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if extra_body:
        llm_kwargs["extra_body"] = extra_body
    return ChatOpenAI(**llm_kwargs)


def make_llm_node(
    llm: Any,
    *,
    prompt_text: str,
    make_inputs: Callable,
    parser_output=None,
    state_key: str = "history",
    state_append: bool = True,
    node_name: str = "NODE",
    printout: bool = True,
    skip_parser: bool = False,
) -> StateCallable:
    prompt = PromptTemplate.from_template(prompt_text)
    parser: Any | None = None
    format_instructions = ""
    if not skip_parser:
        if parser_output is not None:
            parser = PydanticOutputParser(pydantic_object=parser_output)
            format_instructions = parser.get_format_instructions()
        else:
            parser = StrOutputParser()
    chain = prompt | llm
    if parser is not None:
        chain = chain | parser

    def node(state):
        logger.info(f"============= {node_name} ==============")
        inputs = make_inputs(state)
        if format_instructions:
            inputs["format_instructions"] = format_instructions

        result = chain.invoke(inputs)

        if isinstance(parser, PydanticOutputParser):
            result = result.model_dump()

        if state_append:
            state[state_key].append(result)
        else:
            state[state_key] = result

        if printout:
            logger.info(f"AI Answer:\n{result}\n")

        return state

    return node


def _resolve_model_enum(model_name: ModelNames | str) -> ModelNames:
    if isinstance(model_name, ModelNames):
        return model_name
    try:
        return ModelNames(model_name)
    except ValueError:
        return ModelNames[model_name]


def create_graph(config):
    goal_llm = create_llm(
        model_name=_resolve_model_enum(config.runner.goal_decomp_node.model_name),
        prompt_cache_key=config.runner.goal_decomp_node.prompt_cache_key,
    )
    task_llm = create_llm(
        model_name=_resolve_model_enum(config.runner.task_decomp_node.model_name),
        prompt_cache_key=config.runner.task_decomp_node.prompt_cache_key,
    )

    goal_node = make_llm_node(
        llm=goal_llm,
        prompt_text=prompt_module.GOAL_DECOMP_NODE_PROMPT,
        make_inputs=prompt_module.make_goal_decomp_node_inputs,
        parser_output=prompt_module.GoalDecompNodeParser,
        state_key="subgoals",
        state_append=False,
        node_name="GOAL_DECOMP_NODE",
    )
    task_node = make_llm_node(
        llm=task_llm,
        prompt_text=prompt_module.TASK_DECOMP_NODE_PROMPT,
        make_inputs=prompt_module.make_task_decomp_node_inputs,
        parser_output=prompt_module.TaskDecompNodeParser,
        state_key="tasks",
        state_append=False,
        node_name="TASK_DECOMP_NODE",
    )

    workflow = StateGraph(state_schema=StateSchema)
    workflow.add_node("goal_decomp", goal_node)
    workflow.add_node("task_decomp", task_node)

    workflow.add_edge(START, "goal_decomp")
    workflow.add_edge("goal_decomp", "task_decomp")
    workflow.add_edge("task_decomp", END)

    graph = workflow.compile(checkpointer=None)
    return graph
