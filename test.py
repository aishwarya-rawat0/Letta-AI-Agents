from letta import create_client
from letta.schemas.memory import ChatMemory
from letta import EmbeddingConfig, LLMConfig
import logging
from letta.schemas.llm_config import LLMConfig
import os
from letta.prompts import gpt_system
from dotenv import load_dotenv

load_dotenv()

#configure logging 
logging.basicConfig(level=logging.INFO, force=True)

logger = logging.getLogger(__name__)

    
class AgentManager:
    def __init__(self,agent_name):
        self.agent_name = agent_name
        self.client = create_client()
        self.agent_state = None
        self.logger = logging.getLogger(__name__)
        
    def delete_agent_if_exists(self):
        if self.client.get_agent_id(self.agent_name):
            self.client.delete_agent(self.client.get_agent_id(self.agent_name))   
             
    def create_agent(self):
        "to create an agent "
        self.agent_state = self.client.create_agent(
            name = self.agent_name,
            memory = ChatMemory(
                human ="My name is aish",
                persona = "Helpful assistant who loves emojis"
            ),
            
            llm_config=LLMConfig(
            model='llama3.2',
            model_endpoint_type="ollama",
            model_endpoint='http://localhost:11434',
            context_window=8000,
            ),
            embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_model="text-embedding-ada-002",
            embedding_dim=1536,
            embedding_chunk_size=300
            ),
            system=gpt_system.get_system_text('memgpt_chat'),
            include_base_tools=True,
            tools=[] ,  
            agent_type='memgpt_agent',         
            
        )
        
        self.logger.info(f"Created agent with name {self.agent_state.name} and unique ID {self.agent_state.id}")
        
    def list_agents(self):
        "to check the agents we have "
        agents = self.client.list_agents()
        self.logger.info(f"agents:{agents}")
        return agents
    
    
    def send_message(self, message , role = 'user'):
        "message send by the user which is memory of the human to the agent "
        if self.agent_state is None:
            self.logger.error("Agent not created yet")
            return None
        response = self.client.send_message(
            agent_id = self.agent_state.id,
            message = message,
            role = role
        )
                # Extract the assistant's final reply
        assistant_reply = next(
            (msg.message for msg in response.messages if msg.message_type == 'assistant'), 
            None
        )
        if assistant_reply:
            self.logger.info(f"Agent replied: {assistant_reply}")
        else:
            self.logger.warning("No assistant reply found.")
        return assistant_reply

    def get_agent_tools(self):
        """Get tools associated with the agent."""
        if self.agent_state is None:
            self.logger.error("Agent not created yet.")
            return None
        tools = self.agent_state.tools
        self.logger.info(f"Agent tools: {tools}")
        return tools
 
