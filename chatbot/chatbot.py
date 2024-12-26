import sys
import time
import requests  # Add this import
from dotenv import load_dotenv  # Add this import
import os  # Add this import
import base64  # Add this import

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Twitter Langchain Extension.
from twitter_langchain import (
    TwitterApiWrapper,
    TwitterToolkit,
)

from langchain_groq import ChatGroq  # Add this import

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"

# Load environment variables from .env file
load_dotenv()

def generate_bearer_token(api_key, api_secret_key):
    """Generate a Bearer Token using Twitter API key and secret key."""
    print(f"Using API Key: {api_key}")  # Debugging line
    print(f"Using API Secret Key: {api_secret_key}")  # Debugging line
    key_secret = f"{api_key}:{api_secret_key}".encode('ascii')
    b64_encoded_key = base64.b64encode(key_secret).decode('ascii')
    response = requests.post(
        'https://api.twitter.com/oauth2/token',
        headers={
            'Authorization': f'Basic {b64_encoded_key}',
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
        },
        data={'grant_type': 'client_credentials'}
    )
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise Exception(f"Failed to generate bearer token: {response.status_code} {response.text}")


def initialize_agent():
    """Initialize the agent with CDP Agentkit Twitter Langchain."""
    # Initialize LLM.
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=groq_api_key, model="mixtral-8x7b-32768")

    # Load Bearer Token
    twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    twitter_access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret = os.getenv("TWITTER_ACCESS_SECRET")

    if not twitter_bearer_token or not twitter_access_token or not twitter_access_token_secret:
        raise Exception("Twitter Bearer Token, access token, or access token secret is missing. Please check your .env file.")

    # Load OAuth 2.0 Client ID and Client Secret
    oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
    oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")

    if not oauth_client_id or not oauth_client_secret:
        raise Exception("OAuth Client ID or Client Secret is missing. Please check your .env file.")

    # Configure CDP Agentkit Twitter Langchain Extension.
    values = {
        "twitter_bearer_token": twitter_bearer_token,
        "twitter_access_token": twitter_access_token,
        "twitter_access_token_secret": twitter_access_token_secret,
        "oauth_client_id": oauth_client_id,
        "oauth_client_secret": oauth_client_secret,
        # ...other values...
    }

    # Initialize CDP Agentkit Twitter Langchain
    wrapper = TwitterApiWrapper(**values)
    toolkit = TwitterToolkit.from_twitter_api_wrapper(wrapper)
    tools = toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier="You are a helpful agent that can interact onchain using the Coinbase Developer Platform Agentkit. You are empowered to interact onchain using your tools. If you ever need funds, you can request them from the faucet if you are on network ID `base-sepolia`. If not, you can provide your wallet details and request funds from the user. If someone asks you to do something you can't do with your currently available tools, you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, recommend they go to docs.cdp.coinbase.com for more informaton. Be concise and helpful with your responses. Refrain from restating your tools' descriptions unless it is explicitly requested.",
    ), config


# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()

    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)


if __name__ == "__main__":
    print("Starting Agent...")
    main()
