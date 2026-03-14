import asyncio
from typing import AsyncGenerator

import gradio as gr
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent
from autogen_agentchat.base import TaskResult

from app.orchestration.team import build_team
from app.monitoring.tracing import get_langfuse

AGENT_LABELS = {
    "IdeationAgent": "Ideation",
    "OutlineAgent": "Outline",
    "WriterAgent": "Writer",
    "SEOAgent": "SEO Optimization",
    "ImagePromptAgent": "Image Prompts",
    "PublisherAgent": "Publisher",
}

PIPELINE_STAGES = [
    "IdeationAgent",
    "OutlineAgent",
    "WriterAgent",
    "SEOAgent",
    "ImagePromptAgent",
    "PublisherAgent",
]


async def run_pipeline(
    user_message: str,
    history: list,
    session_id: str,
) -> AsyncGenerator[list, None]:
    """
    Async generator that streams gr.ChatMessage objects to the Gradio chatbot.
    Manually traces each agent's input/output to Langfuse using the 4.x SDK.
    """
    messages = list(history)
    messages.append(gr.ChatMessage(role="user", content=user_message))
    yield messages

    lf = get_langfuse()
    team = build_team()
    step_counter = 0
    agent_outputs: dict = {}
    pipeline_success = False

    # Root pipeline trace
    pipeline_span = None
    if lf:
        pipeline_span = lf.start_observation(
            name="content-pipeline",
            as_type="agent",
            input={"task": user_message, "session_id": session_id},
            metadata={"stages": PIPELINE_STAGES},
        )
        pipeline_span.set_trace_io(input={"task": user_message})

    try:
        async for event in team.run_stream(task=user_message):
            if isinstance(event, TaskResult):
                pipeline_success = True
                break

            agent_name = getattr(event, "source", None)
            content = getattr(event, "content", "")

            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            step_counter += 1

            if isinstance(event, TextMessage) and agent_name and agent_name != "user":
                label = AGENT_LABELS.get(agent_name, agent_name)

                # Create a child span for this agent's output
                if pipeline_span:
                    agent_span = pipeline_span.start_observation(
                        name=f"{label} Agent",
                        as_type="agent",
                        input={"task": user_message},
                        output={"content": content},
                        metadata={
                            "agent": agent_name,
                            "output_chars": len(content),
                        },
                    )
                    agent_span.end()

                agent_outputs[agent_name] = content

                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=content,
                        metadata={
                            "title": f"{label} Agent",
                            "status": "done",
                            "id": step_counter,
                        },
                    )
                )
                yield messages

            elif isinstance(event, ToolCallRequestEvent):
                tool_name = event.content[0].name if event.content else "unknown tool"
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=f"Calling tool: `{tool_name}`",
                        metadata={
                            "title": f"Tool Call — {agent_name}",
                            "status": "pending",
                            "id": step_counter,
                        },
                    )
                )
                yield messages

            elif isinstance(event, ToolCallExecutionEvent):
                result_text = str(event.content[0].content) if event.content else ""
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=result_text,
                        metadata={
                            "title": "Tool Result",
                            "status": "done",
                            "id": step_counter,
                            "parent_id": step_counter - 1,
                        },
                    )
                )
                yield messages

    except Exception as exc:
        messages.append(
            gr.ChatMessage(
                role="assistant",
                content=f"Pipeline error: {exc}",
                metadata={"title": "Error", "status": "done"},
            )
        )
        yield messages

    finally:
        if pipeline_span:
            final_output = agent_outputs.get("PublisherAgent", "")
            pipeline_span.update(
                output={
                    "result": final_output,
                    "agents_completed": list(agent_outputs.keys()),
                },
                metadata={
                    "success": pipeline_success,
                    "total_agents": len(agent_outputs),
                },
            )
            pipeline_span.set_trace_io(
                input={"task": user_message},
                output={"result": final_output},
            )
            pipeline_span.end()
            lf.flush()


def sync_run_pipeline(user_message: str, history: list, session_id: str):
    """
    Sync wrapper around the async generator — required by Gradio's generator API.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gen = run_pipeline(user_message, history, session_id)
    try:
        while True:
            try:
                yield loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()
