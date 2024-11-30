# Browser Agent

A browser automation agent built with Python, utilizing LangChain and LangGraph frameworks. This agent can perform autonomous web browsing tasks through a series of coordinated actions.

## Overview

This browser agent implements a flow-based architecture that allows it to perform various browser interactions including:
- Clicking elements
- Navigating back
- Performing Google searches
- Scrolling pages
- Typing text
- Waiting for elements/pages to load

## Architecture

The agent uses a graph-based workflow with the following key actions:
- `_start`: Entry point for the agent workflow
- `Click`: Handles element clicking operations
- `GoBack`: Manages browser back navigation
- `Google`: Performs Google search operations
- `Scroll`: Controls page scrolling
- `Type`: Handles text input
- `Wait`: Manages waiting operations
- `update_scratchpad`: Updates the agent's memory/context
- `agent`: Core decision-making component

## Prerequisites

- Python 3.8+
- LangChain
- LangGraph
- Selenium WebDriver or Playwright (for browser automation)

## Installation

```bash
pip install langchain langgraph selenium  # or playwright
```

## Usage

Basic example of initializing and running the agent:

```python
from langchain import BrowserAgent
from langgraph import Graph

# Initialize the agent
browser_agent = BrowserAgent()

# Define the workflow
workflow = Graph()
workflow.add_node("_start", browser_agent.start)
workflow.add_node("Click", browser_agent.click)
# Add other nodes...

# Run the agent
result = workflow.run()
```

## Flow Diagram

The agent follows a directed graph workflow where each node represents a specific browser operation. The flow begins at `_start` and transitions between different actions based on the agent's decision-making logic and the current state of the webpage.

## Contributing

Feel free to submit issues and enhancement requests.

## License
