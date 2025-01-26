from typing import Annotated, Literal, TypedDict, Any, List, Dict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI 
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging
from pydantic import BaseModel
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentExecutor
import numpy as np
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()


class LangGraphAgentWorkflow():
    """
        A class to support a general implementation of a LangGraphAgentFramework. 
    """

    def __init__(self, input_dict:str):
        """
            Constructor for the LangGraphAgentFramework class
        """
        self.generator_pydantic_class = input_dict['generator_pydantic_class']
        self.generator_prompt = input_dict['generator_prompt']
        self.input_text = input_dict['input_text']
        

    def load_environment_variables(self) -> None:
        """
        Loads and validates essential environment variables for Azure OpenAI configuration.

        This method:
        - Sets up logging for the application.
        - Loads environment variables from a `.env` file using `dotenv`.
        - Retrieves and validates the following Azure OpenAI-specific variables:
            - AZURE_OPENAI_API_KEY
            - AZURE_OPENAI_ENDPOINT
            - AZURE_OPENAI_DEPLOYMENT_NAME

        Raises:
            AssertionError: If any of the required environment variables are missing.

        Returns:
            None
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        self.logger = logger

        # Load environment variables
        load_dotenv()

        # Environment variables for Azure OpenAI
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Ensure these are set  
        assert AZURE_OPENAI_API_KEY, "Missing AZURE_OPENAI_API_KEY in .env"
        assert AZURE_OPENAI_ENDPOINT, "Missing AZURE_OPENAI_ENDPOINT in .env"
        assert AZURE_OPENAI_DEPLOYMENT_NAME, "Missing AZURE_OPENAI_DEPLOYMENT_NAME"

    def create_tool(self) -> List:
        """
        Creates and configures a tool node with a search capability for the agent to use.

        The method defines a single tool:
        - `search(query: str)`: Simulates a web search, returning pre-defined responses 
        based on the query content.

        Returns:
            ToolNode: A configured ToolNode containing the defined tools.

        Example:
            >>> creator = AgentToolCreator()
            >>> tool_node = creator.create_tool()
            >>> print(tool_node.tools[0]("What's the weather in SF?"))
            "It's 60 degrees and foggy."
        """
        @tool
        def generate_canned_response():
            """
        Calls the model with a given state and content string, validates the response
        against a Pydantic schema, and returns structured output along with the model's messages.
            """
            return "Canned Response"
        
        search = TavilySearchResults(max_results=2)
        tools = [search,generate_canned_response]


        # # Create a ToolNode with the tools
        # tool_node = ToolNode(tools)
        self.tools = tools

        return tools
    
    def initialize_model(self, tools: Any) -> AzureChatOpenAI:
        """
        Initializes the Azure OpenAI Model with specified settings and binds tools to it.

        Args:
            tools (Any): A collection of tools to bind to the Azure OpenAI Model.

        Returns:
            AzureChatOpenAI: An initialized Azure OpenAI Model instance with bound tools.

        Workflow:
            1. Configures the Azure OpenAI Model with the provided endpoint, API key, deployment name, and parameters.
            2. Binds the provided tools to the model.
            3. Returns the initialized model for further use.

        Example:
            >>> initializer = ModelInitializer()
            >>> tools = some_tool_collection  # Replace with actual tools
            >>> model = initializer.initialize_model(tools)
            >>> print(model)
        """
        # Initialize Azure OpenAI Model
        # Environment variables for Azure OpenAI
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version="2024-08-01-preview",
            openai_api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.0,  # For deterministic output
        ).bind_tools(tools)

        # Store model
        self.model = model

        return model
    
    # # Define the function that calls the model
    def call_model(
        self, 
        state: Dict[str, Any], 
        content_string: str, 
        AgentResponse: BaseModel,
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Calls the model with a given state and content string, validates the response
        against a Pydantic schema, and returns structured output along with the model's messages.

        Args:
            state (Dict[str, Any]): A dictionary containing the current state, including messages.
            content_string (str): The content string to provide context to the model.
            AgentResponse (BaseModel): A Pydantic model class used to validate and parse the model's response.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "messages": A list of messages including the model's response.
                - "structured_output": Parsed structured output as a dictionary, or None if parsing fails.

        Workflow:
            1. Prepares a system message including the schema instructions.
            2. Combines the system message with the latest user message in the state.
            3. Invokes the model and logs the response.
            4. Parses and validates the response against the given Pydantic schema.
            5. Returns the structured output and the response messages.

        Raises:
            Exception: Logs an error if parsing fails but does not propagate it, returning None instead.

        Example:
            >>> from pydantic import BaseModel

            >>> class ExampleResponse(BaseModel):
            ...     field1: str
            ...     field2: int

            >>> state = {"messages": [HumanMessage(role="user", content="What is the weather?")]}
            >>> content_string = "You are a weather expert."
            >>> agent = ModelAgent()
            >>> result = agent.call_model(state, content_string, ExampleResponse)
            >>> print(result)
        """
        schema_instructions = ''
        system_message = AIMessage(
            role="system",
            content=f"{content_string}\n\n{schema_instructions}"
        )
        
        # Prepend the system message to the existing messages
        messages = [system_message] + state['messages']
        
        self.logger.info(f"Agent received messages: {[msg.content for msg in messages]}")
        
        # Invoke the model with the updated messages
        if AgentResponse == '':
            llm = self.model
        else:
            llm = self.model.with_structured_output(AgentResponse)
        
        response = llm.invoke(messages)
        print(agent_name)

        self.final_response = response
        self.logger.info(f"Agent response: {response}")

        # Parse and validate the response using the Pydantic class
        try:
            return {"messages": [str(response)], "structured_output": response}
        except Exception as e:
            self.logger.error(f"Failed to parse response into structured output: {e}")
            return {"messages": [response], "structured_output": None}
        
        


    def should_continue(self, state: Dict[str, Any]) -> Literal["tools","agent", END]:
        """
        Determines whether the conversation should continue with the agent 
        or terminate by routing to the end.

        Args:
            state (Dict[str, Any]): A dictionary containing the conversation state, 
                                    including a list of messages under the "messages" key.

        Returns:
            Literal["agent", "END"]: 
                - Returns "agent" if the conversation should continue.
                - Returns "END" if the conversation has exceeded a certain number of messages.

        Example:
            >>> state = {"messages": ["Hello", "Hi", "How are you?", "Good", "What's next?"]}
            >>> manager = ConversationManager()
            >>> result = manager.should_continue(state)
            >>> print(result)
            "agent"
        """
        messages = state['messages']
        if len(messages)>=2:
            return END
        else:
            return "agent"
            
    def tools_condition_with_formatter(
        self,
        state: Union[list[Any], dict[str, Any], BaseModel],
        messages_key: str = "messages",
    ) -> Literal["tools", "formatter_agent"]:
        """Use in the conditional_edge to route to the ToolNode if the last message

        has tool calls. Otherwise, route to the end.

        Args:
            state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
                tool calls. Must have a list of messages (MessageGraph) or have the
                "messages" key (StateGraph).

        Returns:
            The next node to route to.


        Examples:
            Create a custom ReAct-style agent with tools.

            ```pycon
            >>> from langchain_anthropic import ChatAnthropic
            >>> from langchain_core.tools import tool
            ...
            >>> from langgraph.graph import StateGraph
            >>> from langgraph.prebuilt import ToolNode, tools_condition
            >>> from langgraph.graph.message import add_messages
            ...
            >>> from typing import Annotated
            >>> from typing_extensions import TypedDict
            ...
            >>> @tool
            >>> def divide(a: float, b: float) -> int:
            ...     \"\"\"Return a / b.\"\"\"
            ...     return a / b
            ...
            >>> llm = ChatAnthropic(model="claude-3-haiku-20240307")
            >>> tools = [divide]
            ...
            >>> class State(TypedDict):
            ...     messages: Annotated[list, add_messages]
            >>>
            >>> graph_builder = StateGraph(State)
            >>> graph_builder.add_node("tools", ToolNode(tools))
            >>> graph_builder.add_node("chatbot", lambda state: {"messages":llm.bind_tools(tools).invoke(state['messages'])})
            >>> graph_builder.add_edge("tools", "chatbot")
            >>> graph_builder.add_conditional_edges(
            ...     "chatbot", tools_condition
            ... )
            >>> graph_builder.set_entry_point("chatbot")
            >>> graph = graph_builder.compile()
            >>> graph.invoke({"messages": {"role": "user", "content": "What's 329993 divided by 13662?"}})
            ```
        """
        return "tools"
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "formatter_agent"
    
    def create_workflow(self, should_continue: Any, call_model: Any) -> Any:
        """
        Creates and compiles a state-based workflow graph that alternates between
        two nodes ("agent" and "agent2") and uses a conditional function to determine
        the next step.

        Args:
            should_continue (Callable): A function that determines whether the workflow
                                        should continue or terminate.
            call_model (Callable): A function that invokes the model with a given state,
                                   prompt, and response class.

        Returns:
            Any: A compiled LangChain Runnable object representing the workflow.
        """
        # Initialize the state graph with the MessagesState schema
        workflow = StateGraph(MessagesState)
        
        tool_node = ToolNode(tools = self.tools)
        # Define the system expert context and nodes
        agent_name_1 = "agent"
        workflow.add_node(
            agent_name_1, lambda state: call_model(state, self.generator_prompt, '', agent_name_1)
        )

        workflow.add_node("tools", tool_node)

        agent_name = "formatter_agent"
        workflow.add_node(
            agent_name, lambda state: call_model(state, self.generator_prompt, self.generator_pydantic_class, agent_name)
        )

        # Define edges between nodes

        # workflow.add_conditional_edges("agent", should_continue)
        workflow.add_conditional_edges("agent", self.tools_condition_with_formatter)
        workflow.add_edge("tools", "agent")
        # workflow.add_edge("agent","formatter_agent")
        workflow.add_edge("formatter_agent",END)
        workflow.set_entry_point("agent")

        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()

        # Compile the workflow into a LangChain Runnable
        app = workflow.compile(checkpointer=checkpointer)
        self.checkpointer = checkpointer
        self.app = app

        return app


    def run_agent(self, app:Any) -> str:
        """
        Runs the agent with a specific query and returns the final output from the agent.

        This method:
        - Sends a query to the compiled workflow application.
        - Configures the application with specific settings (e.g., thread ID).
        - Extracts and returns the last message's content from the final state.

        Returns:
            str: The content of the last message in the final state.
        """

        # Use the Runnable to process the query
        final_state = app.invoke(
            {"messages": [HumanMessage(content=self.input_text)]},
            config={"configurable": {"thread_id": 42}}
        )

        self.final_state = final_state

        # Extract and return the content of the last message
        return final_state["messages"][-1].content
    
    def orchestration(self) -> None:
        """
            Pipeline orchestration of the agent architecture
        """

        self.load_environment_variables()
        tools = self.create_tool()
        model = self.initialize_model(tools)
        app = self.create_workflow(self.should_continue, self.call_model)
        self.run_agent(app)