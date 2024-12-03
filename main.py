import os
import uuid
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import traceback
import json

from letta import create_client
from letta.schemas.memory import ChatMemory
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.config import LettaConfig
from letta.agent_store.storage import StorageConnector, TableType
from letta.prompts import gpt_system
from letta.schemas.message import Message
from letta.schemas.enums import MessageRole
from letta.schemas.openai.chat_completions import ToolCall

from neo4j import GraphDatabase
from dotenv import load_dotenv
from letta.services.organization_manager import OrganizationManager
from letta.schemas.user import UserCreate


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API Key is not set or found in the environment variables.")

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Simplified Neo4jStorageConnector
class Neo4jStorageConnector(StorageConnector):
    """Storage via Neo4j implementation"""

    def __init__(self, table_type: str, config: LettaConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.session = self.driver.session()
            logger.debug("Connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            traceback.print_exc()
            raise

    def get_filters(self, filters: Optional[Dict] = None) -> str:
        """Implement proper filter generation for Neo4j queries"""
        if not filters:
            return ""
        # Implement Neo4j-specific filter string generation
        return ""

    def create(self, data: Dict):
        """Implement create operation"""
        pass

    def read(self, identifier: str):
        """Implement read operation"""
        pass

    def update(self, identifier: str, data: Dict):
        """Implement update operation"""
        pass

    def delete(self, identifier: str):
        """Implement delete operation"""
        pass

    def list(self, cursor: Optional[str] = None, limit: Optional[int] = None):
        """Implement list operation"""
        pass


# Update the StorageConnector to include our Neo4jStorageConnector
def get_storage_connector(
    table_type: str,
    config: LettaConfig,
    user_id,
    agent_id=None,
):
    storage_type = 'neo4j'  # For testing, we set this directly

    if storage_type == "neo4j":
        return Neo4jStorageConnector(table_type, config, user_id, agent_id)
    else:
        raise NotImplementedError(f"Storage type {storage_type} not implemented")

# Monkey patch the StorageConnector's get_storage_connector method
StorageConnector.get_storage_connector = staticmethod(get_storage_connector)

# Main code
def main():
    print("Starting the main function.")

    # Define the agent name
    agent_name = "neo4j_test_agent"
    print(f"Agent name: {agent_name}")

    # Create a Letta client
    print("Creating Letta client...")
    client = create_client()
    
    # User management
    user_name = "test_user"
    user_id = None

    try:
        print(f"Checking if user '{user_name}' exists...")
        # Get list of users
        users = client.server.user_manager.list_users()
        user = next((u for u in users if u.name == user_name), None)
        if user:
            user_id = user.id
            print(f"User '{user_name}' found with ID '{user_id}'.")
        else:
            print(f"User '{user_name}' not found. Creating new user.")
            user_create = UserCreate(name=user_name, organization_id=OrganizationManager.DEFAULT_ORG_ID)
            user = client.server.user_manager.create_user(pydantic_user=user_create)
            user_id = user.id
            print(f"User '{user_name}' created with ID '{user_id}'.")
    except Exception as e:
        print(f"Failed to create or retrieve user '{user_name}': {e}")
        raise

    # Assign the user_id to the client
    client.user_id = user_id
    print(f"Letta client created with user ID: {client.user_id}")

    # Set LLM and Embedding configurations
    print("Setting default LLM and embedding configurations...")
    client.set_default_llm_config(LLMConfig.default_config(model_name='gpt-4o-mini'))
    client.set_default_embedding_config(EmbeddingConfig.default_config(model_name='text-embedding-ada-002'))
    print("Configurations set.")

    # Check if the agent exists and delete if it does
    try:
        print(f"Checking if agent '{agent_name}' exists...")
        existing_agent_id = client.get_agent_id(agent_name)
        if existing_agent_id:
            print(f"Deleting existing agent '{agent_name}' with ID '{existing_agent_id}'")
            client.delete_agent(existing_agent_id)
            logger.info(f"Deleted existing agent '{agent_name}' with ID '{existing_agent_id}'")
    except Exception as e:
        print(f"Error while checking or deleting agent: {e}")
        traceback.print_exc()

    # Create a new agent
    try:
        print("Creating a new agent...")
        agent_state = client.create_agent(
            name=agent_name,
            memory=ChatMemory(
                human="My name is Aishwarya",
                persona="You are a helpful assistant."
            ),
            system=gpt_system.get_system_text('memgpt_chat')
        )
        agent_id = agent_state.id
        print(f"Agent created with ID: {agent_id}")
        print(f"Agent created with userID: {agent_state.user_id}")

        logger.info(f"Created agent with name '{agent_state.name}' and unique ID '{agent_state.id}'")

        # Create an instance of Neo4jStorageConnector for testing
        print("Creating Neo4jStorageConnector...")
        storage_connector = Neo4jStorageConnector(
            table_type=TableType.RECALL_MEMORY,
            config=LettaConfig.load(),
            user_id=client.user_id,
            agent_id=agent_id
        )
        print("Neo4jStorageConnector created.")

        # **Test the insert and get methods**
        print("Testing insert and get methods...")
        # Create a test message
        test_message = Message(
            id=str(uuid.uuid4()),
            role=MessageRole.user,
            text="Test message",
            user_id=client.user_id,
            agent_id=agent_id,
            created_at=datetime.utcnow()
        )

        # Insert the message
        storage_connector.insert(test_message)

        # Retrieve the message
        retrieved_message = storage_connector.get(test_message.id)

        # Assert that the message was retrieved correctly
        assert retrieved_message is not None, "Retrieved message is None"
        assert retrieved_message.text == test_message.text, "Retrieved message text does not match"

        logger.info("Test message inserted and retrieved successfully.")
        print("Test message inserted and retrieved successfully.")

        # Interact with the agent
        print("Sending message to agent: 'Hello!'")
        response = client.send_message(
            agent_id=agent_state.id,
            message="Hello!",
            role="user"
        )
        logger.info(f"Agent response to 'Hello!': {response.messages[-1].content}")
        print(f"Agent response to 'Hello!': {response.messages[-1].content}")

        # Insert a memory into archival memory
        print("Inserting a memory into archival memory...")
        archival_response = client.send_message(
            agent_id=agent_state.id,
            message="Remember that I love chocolate.",
            role="user"
        )
        logger.info(f"Archival memory response: {archival_response.messages[-1].content}")
        print(f"Archival memory response: {archival_response.messages[-1].content}")

        # Retrieve memory from archival memory
        print("Retrieving memory from archival memory...")
        retrieval_response = client.send_message(
            agent_id=agent_state.id,
            message="What do I love?",
            role="user"
        )
        logger.info(f"Memory retrieval response: {retrieval_response.messages[-1].content}")
        print(f"Memory retrieval response: {retrieval_response.messages[-1].content}")

    except Exception as e:
        logger.error(f"An error occurred while creating or interacting with the agent: {e}")
        traceback.print_exc()

    print("Main function completed.")

if __name__ == "__main__":
    main()