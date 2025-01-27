from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Any, List, Dict, Set, Union
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.callback import CallbackHandler



class LangGraphAgentSystem():
    """
        A class to support a minimal implementation of a LangGraph Agentic Framework.

        This class implements a minimal agentic framework. 
        A single agent with a tool that routes to another downstream agent that does pydantic structured output.
    """

    def __init__(self, input_dict:dict):
        """
            Constructor for the LangGraphAgentFramework class
        """
        self.input_dict = input_dict

    def initialize_model(self):
        """
        Initializes the Azure OpenAI model with Langfuse monitoring and tools binding.

        This method performs the following tasks:
        1. Loads environment variables using `dotenv`.
        2. Initializes the Langfuse client and callback handler for monitoring and logging.
        3. Configures the Azure OpenAI model with the API key, endpoint, and deployment name.
        4. Binds tools to the model and sets up structured output handling.

        Attributes set:
        - `self.langfuse`: An instance of the Langfuse client for monitoring.
        - `self.langfuse_handler`: A Langfuse callback handler for interaction logging.
        - `self.model_with_tools`: The Azure OpenAI model with tools bound to it.
        - `self.model_with_structured_output`: The Azure OpenAI model with structured output handling.

        Raises:
            KeyError: If any required environment variable is not found.
            Exception: If there is an error during model initialization.

        Returns:
            None
        """
        # Load environment variables
        load_dotenv()

        # Initialize Langfuse client
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )

        # Initialize Langfuse callback handler
        self.langfuse_handler = CallbackHandler(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )

        # Load Azure OpenAI API credentials
        AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

        # Validate required credentials
        if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT_NAME):
            raise KeyError("Missing Azure OpenAI API credentials in environment variables.")

        # Initialize the Azure OpenAI model
        model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version="2024-08-01-preview",
            openai_api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.0,  # For deterministic output
        )

        # Bind tools to the model
        model_with_tools = model.bind_tools(self.tools)

        # Set up structured output
        model_with_structured_output = model.with_structured_output(self.input_dict['structured_output_class'])

        # Assign models to instance attributes
        self.model_with_tools = model_with_tools
        self.model_with_structured_output = model_with_structured_output

    
    def create_tools(self) -> None:
        """
        Creates and registers tools for the agent.

        This method defines tools that the agent can use to perform specific tasks. 
        In this case, it defines a tool to retrieve weather information for specific cities.

        Attributes:
            self.tools (List[Callable]): A list of tools registered for the agent.

        Tools:
            - `get_weather`: A tool to retrieve weather information for NYC or SF.

        Raises:
            AssertionError: If the `city` parameter is not one of the expected values.

        Returns:
            None
        """
        @tool
        def get_weather(city: Literal["nyc", "sf"]) -> str:
            """Use this to get weather information."""
            if city == "nyc":
                return "It is cloudy in NYC, with 5 mph winds in the North-East direction and a temperature of 70 degrees"
            elif city == "sf":
                return "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"
            else:
                raise AssertionError("Unknown city")
        
        # Register tools
        tools = [get_weather]
        self.tools = tools


    def call_model(self, state: MessagesState) -> Dict[str, List[SystemMessage]]:
        """
        Calls the model with the provided state and appends the system message.

        This method appends a system message containing the agent's prompt to the existing message state,
        invokes the model to generate a response, and returns the updated state.

        Parameters:
            state (MessagesState): The current state of the conversation, including a list of messages.

        Returns:
            Dict[str, List[SystemMessage]]: A dictionary containing the updated messages, 
            including the model's response.

        Attributes:
            self.input_dict (Dict): A dictionary containing configurations, including the agent's prompt.
            self.model_with_tools: The model instance with tools bound, used for generating responses.

        Example:
            state = {
                "messages": [HumanMessage(content="What's the weather in SF?")]
            }
            result = call_model(state)
            print(result["messages"])
        """
        # Append the agent's system message to the conversation
        state['messages'].append(SystemMessage(
            content=self.input_dict["agent_prompt"]
        ))

        # Call the model with the updated message state
        response = self.model_with_tools.invoke(state["messages"])
        
        # Return the updated messages as a list
        return {"messages": [response]}



    def respond(self, state: MessagesState) -> Dict[str, Union[str, HumanMessage]]:
        """
        Responds to the user by processing the last tool message and invoking the model
        with structured output to maintain consistent formatting.

        This method checks if the third-to-last message in the state is a `ToolMessage`.
        If so, it converts the tool's output into a `HumanMessage` and invokes the model 
        to generate a structured response. If no tool message is found, it exits gracefully.

        Parameters:
            state (MessagesState): The current state of the conversation, including a list of messages.

        Returns:
            Dict[str, Union[str, HumanMessage]]: A dictionary containing the final response:
                - "final_response": A structured response from the model if a tool was invoked,
                or an exit message if no tool was called.
        """
        # Check if the third-to-last message is a ToolMessage
        if isinstance(state['messages'][-3], ToolMessage):
            # Convert the ToolMessage content to a HumanMessage
            response = self.model_with_structured_output.invoke(
                [HumanMessage(content=state["messages"][-3].content)]
            )
            # Return the final structured response
            return {"final_response": response}
        else:
            # Exit gracefully if no ToolMessage is found
            print("Tool was not called")
            return {"final_response": "Exit: tool not called"}


    def should_continue(self, state: MessagesState) -> Literal["respond", "continue"]:
        """
        Determines whether to continue the workflow or respond to the user based on the last message in the conversation.

        This method checks the last message in the conversation state. If the message contains tool calls,
        it indicates that the workflow should continue. Otherwise, it decides to respond to the user.

        Parameters:
            state (MessagesState): The current state of the conversation, including a list of messages.

        Returns:
            Literal["respond", "continue"]: A string indicating the next action:
                - "respond": If there are no tool calls in the last message, respond to the user.
                - "continue": If the last message contains tool calls, continue the workflow.
        """
        # Extract the list of messages and the last message
        messages = state["messages"]
        last_message = messages[-1]

        # Determine whether to continue or respond based on tool calls
        if not last_message.tool_calls:
            return "respond"
        else:
            return "continue"
            
    def create_workflow(self) -> Any:
        """
        Creates and executes a workflow for managing the conversation flow between an agent, tool interactions, 
        and user responses.

        This method defines a `StateGraph` to handle the conversation workflow, including conditional transitions
        based on whether the agent should continue processing or respond to the user. It compiles the workflow 
        and invokes it with the input message, returning the final result.

        Workflow Overview:
            - Nodes:
                - "agent": Handles model invocation using `call_model`.
                - "respond": Handles user responses using `respond`.
                - "tools": Handles tool interactions using a `ToolNode`.
            - Entry Point: "agent"
            - Transitions:
                - From "agent":
                    - If `should_continue` returns "continue", transitions to "tools".
                    - If `should_continue` returns "respond", transitions to "respond".
                - From "tools", transitions back to "agent".
                - From "respond", ends the workflow (`END`).

        Parameters:
            None

        Returns:
            Any: The output of the workflow after invoking the graph, typically the final response.
        """
        # Define a new graph
        workflow = StateGraph(self.input_dict['agent_state'])

        # Define the three nodes in the workflow
        workflow.add_node("agent", self.call_model)
        workflow.add_node("respond", self.respond)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set the entry point to "agent"
        workflow.set_entry_point("agent")

        # Add conditional edges from "agent" based on `should_continue`
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",  # Transition to "tools" if `should_continue` returns "continue"
                "respond": "respond",  # Transition to "respond" if `should_continue` returns "respond"
            },
        )

        # Define other edges
        workflow.add_edge("tools", "agent")  # Cycle back to "agent" from "tools"
        workflow.add_edge("respond", END)   # End the workflow from "respond"

        # Compile the workflow into a graph
        graph = workflow.compile()

        # Invoke the graph with the initial input and callback handler
        answer = graph.invoke(
            input={"messages": [("human", self.input_dict['input_prompt'])]},
            config={"callbacks": [self.langfuse_handler]}
        )

        # Save graph
        self.graph = graph

        return answer


    def run(self) -> Any:
        """
        Executes the main workflow for the agent, including tool creation, model initialization, 
        and workflow execution.

        This method performs the following steps:
        1. Creates tools using `create_tools`.
        2. Initializes the model with monitoring and configuration using `initialize_model`.
        3. Builds and executes the conversation workflow using `create_workflow`.
        4. Stores the final output of the workflow in `self.answer`.

        Returns:
            Any: The final response from the executed workflow.

        Attributes:
            self.answer (Any): Stores the final response from the workflow for future use.

        Raises:
            Exception: If any of the underlying methods (`create_tools`, `initialize_model`, or `create_workflow`)
            encounter an error.

        Example:
            result = run()
            print(result)  # Outputs the final workflow response.
        """
        # Step 1: Create tools
        self.create_tools()

        # Step 2: Initialize the model
        self.initialize_model()

        # Step 3: Create and execute the workflow
        answer = self.create_workflow()

        # Store the final response
        self.answer = answer

        # Return the final response
        return answer
    

