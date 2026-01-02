"""Graph definitions for LangGraph Studio.

Note: LangGraph Studio/API handles persistence automatically.
Don't pass checkpointer/store here - the platform provides these.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import create_react_agent, create_research_agent

# Create agent instances for Studio
# Note: Don't pass checkpointer/store - LangGraph API handles persistence
react_agent = create_react_agent()
research_agent = create_research_agent()
