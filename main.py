import argparse
import asyncio
import base64
import io
import os
import re

import nest_asyncio
from attr import dataclass
from dotenv import load_dotenv
from langchain_core import prompts
from langchain_core.messages import (
    SystemMessage,
    ai,
    chat,
    function,
    human,
    system,
    tool,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import chain as chain_decorator
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image
from playwright.async_api import Page, async_playwright
from playwright_stealth import stealth_async
from langchain_core.runnables.graph import MermaidDrawMethod

# Local import
from agent.browser.state import AgentState
from agent.browser.tools import click, type_text, scroll, wait, go_back, to_URL
from path import Path

load_dotenv()
nest_asyncio.apply()

# 3.Agent:

# 1. `mark_page` function to annotate the current page with a bounding box
# 2. Prompts to hold user questions, annotated images, and agent scratchpads
# 3. GPT-4V deciding next steps
# 4. Parsing logic to extract actions

with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page: Page):
    await page.evaluate(mark_page_script)
    bboxes = []
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


# Agent prompt
async def annotate(state: AgentState) -> dict:
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state: dict) -> dict:
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


# prompt = hub.pull("wfh/web-voyager")
prompt = prompts.ChatPromptTemplate(
    input_variables=["bbox_descriptions", "img", "input"],
    input_types={
        "scratchpad": list[
            ai.AIMessage
            | human.HumanMessage
            | chat.ChatMessage
            | system.SystemMessage
            | function.FunctionMessage
            | tool.ToolMessage
        ]
    },
    partial_variables={"scratchpad": []},
    messages=[
        prompts.SystemMessagePromptTemplate(
            prompt=[
                prompts.PromptTemplate(
                    input_variables=[],
                    template="""
Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will
feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow
the guidelines and choose one of the following actions:

1. Click a Web Element.
2. Delete existing content in a textbox and then type content.
3. Scroll up or down.
4. Wait 
5. Go back
7. Return to provided HomePageUrl
8. Respond with the final answer

Correspondingly, Action should STRICTLY follow the format:

- Click [Numerical_Label] 
- Type [Numerical_Label]; [Content] 
- Scroll [Numerical_Label or WINDOW]; [up or down] 
- Wait 
- GoBack
- Go to HomePageUrl
- ANSWER; [content]

Key Guidelines You MUST follow:

* Action guidelines *
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.

* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
2) Select strategically to minimize time wasted.

Your reply should strictly follow the format:

Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
Action: {{One Action format you choose}}
Then the User will provide:
Observation: {{A labeled screenshot Given by User}}
""",
                )
            ]
        ),
        prompts.MessagesPlaceholder(variable_name="scratchpad", optional=True),
        prompts.HumanMessagePromptTemplate(
            prompt=[
                ImagePromptTemplate(
                    input_variables=["img"],
                    template={"url": "data:image/png;base64,{img}"},
                ),
                prompts.PromptTemplate(
                    input_variables=["bbox_descriptions"],
                    template="{bbox_descriptions}",
                ),
                prompts.PromptTemplate(input_variables=["input"], template="{input}"),
            ]
        ),
    ],
)

llm = ChatOpenAI(model="gpt-4-turbo", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

# 4. Graph:

# agent
# -> update_scratchpad (after a tool is invoked)
# -> click (if prediction.action == "Click")
# -> type_text (if prediction.action == "Type")
# -> scroll (if prediction.action == "Scroll")
# -> wait (if prediction.action == "Wait")
# -> go_back (if prediction.action == "GoBack")
# -> Home_page_url (if prediction.action == "Home_page_url")
# -> select_tool (if prediction.action == "ANSWER" or "retry")
# -> END (if prediction.action == "ANSWER")


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old and isinstance(old[0], str):
        txt = str(old[0])
        last_line = txt.split("\n")[-1]
        ma = re.match(r"\d+", last_line)
        if ma is None:
            txt = "Previous action observations:\n"
            step = 1
        else:
            step = int(ma.group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}


graph_builder = StateGraph(AgentState)

# Nodes (doing the work)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

# Edges (data flow)
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "HomePageUrl": to_URL,
}


for node_name in tools:
    graph_builder.add_node(
        node_name,
        RunnableLambda(tools[node_name])
        | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")


# conditional edge
def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path="agent_graph.png")
# graph.get_graph().print_ascii()


# 5. Run agent
async def call_agent(question: str, page, homeurl, max_steps: int = 500):
    path = Path()
    objective_image = path.create_text_image(
        "Objective: " + question, width=800, height=100, font_size=100
    )
    path.update_agent_path_image(objective_image, is_initial=True)

    event_stream = graph.astream(
        {"page": page, "input": question, "scratchpad": [], "url": homeurl},
        {
            "recursion_limit": max_steps,
        },
    )

    final_answer = None
    steps = []
    step_counter = 0
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")

        step_counter += 1

        steps.append(f"{step_counter}. {action}: {action_input}")
        print(f"{step_counter}. {action}: {action_input}")

        with open("agent_steps.txt", "w") as file:
            file.write("\n".join(steps))

        screenshot_data = base64.b64decode(event["agent"]["img"])
        img = Image.open(io.BytesIO(screenshot_data))
        path.update_agent_path_image(img)

        if action and "ANSWER" in action:
            if action_input is None:
                raise ValueError("No answer provided.")
            final_answer = action_input[0]
            # Create and add the final response image
            final_response_image = path.create_text_image(
                "Final Response: " + final_answer, width=800, height=100, font_size=20
            )
            path.update_agent_path_image(final_response_image, is_final=True)

    return final_answer


# Main Code
async def main():
    # Objective of code
    objective = """You are on a browser agent. Follow these steps:
                - Search for bangalore city.
                - Find good restaurent among those find top 3 rated restaurants.
                - give me details of those  restaurants",
       
                
                """
    url = "https://www.google.com/"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(record_video_dir="videos/")
        page = await context.new_page()
        await stealth_async(page)
        await page.goto("https://www.google.com/")

        try:
            res = await call_agent(objective, page, url)
            print(f"Final response: {res}")
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        exit(1)