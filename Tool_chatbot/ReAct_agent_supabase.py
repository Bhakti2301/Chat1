from __future__ import annotations

import os
import sys
import time
import uuid
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun, SleepTool

try:
    from supabase import create_client, Client
except ImportError:
    print("Error: supabase-py not installed. Install with: pip install supabase")
    sys.exit(1)

class SupabaseMemory:
    """Supabase-based conversation memory for scalable storage."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize Supabase tables if they don't exist."""
        # Note: In Supabase, you typically create tables via the dashboard
        # This is just a check to ensure the table exists
        try:
            # Test connection by trying to select from conversations table
            self.supabase.table('conversations').select('*').limit(1).execute()
            print("✓ Connected to Supabase conversations table")
        except Exception as e:
            print(f"⚠️  Warning: Could not access conversations table: {e}")
            print("Please create the table in Supabase dashboard with this SQL:")
            print("""
            CREATE TABLE conversations (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            -- Create indexes for better performance
            CREATE INDEX idx_conversations_user_session ON conversations(user_id, session_id);
            CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
            """)
    
    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        try:
            data = {
                'user_id': user_id,
                'session_id': session_id,
                'role': role,
                'content': content,
                'timestamp': time.time()
            }
            result = self.supabase.table('conversations').insert(data).execute()
            return result
        except Exception as e:
            print(f"Error adding message to Supabase: {e}")
            return None
    
    def get_recent_messages(self, user_id: str, session_id: str, limit: int = 10):
        """Get recent messages for a user session."""
        try:
            result = self.supabase.table('conversations')\
                .select('role, content, timestamp')\
                .eq('user_id', user_id)\
                .eq('session_id', session_id)\
                .order('timestamp', desc=False)\
                .limit(limit)\
                .execute()
            
            messages = []
            for row in result.data:
                messages.append({
                    "role": row['role'],
                    "content": row['content']
                })
            return messages
        except Exception as e:
            print(f"Error getting messages from Supabase: {e}")
            return []
    
    def get_user_sessions(self, user_id: str):
        """Get all sessions for a user."""
        try:
            result = self.supabase.table('conversations')\
                .select('session_id, timestamp')\
                .eq('user_id', user_id)\
                .order('timestamp', desc=True)\
                .execute()
            
            sessions = {}
            for row in result.data:
                session_id = row['session_id']
                if session_id not in sessions:
                    sessions[session_id] = row['timestamp']
            return sessions
        except Exception as e:
            print(f"Error getting user sessions from Supabase: {e}")
            return {}
    
    def get_all_users(self):
        """Get all unique user IDs."""
        try:
            result = self.supabase.table('conversations')\
                .select('user_id')\
                .execute()
            
            users = set()
            for row in result.data:
                users.add(row['user_id'])
            return list(users)
        except Exception as e:
            print(f"Error getting users from Supabase: {e}")
            return []

def create_supabase_react_agent(user_id: str, session_id: str):
    """Create a ReAct agent with Supabase persistent memory."""
    
    # Check for Supabase credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: Supabase credentials not found!")
        print("Please set these environment variables:")
        print("export SUPABASE_URL='your-supabase-url'")
        print("export SUPABASE_ANON_KEY='your-supabase-anon-key'")
        return None, None, None, None
    
    # Initialize Supabase memory
    supabase_memory = SupabaseMemory(supabase_url, supabase_key)
    
    # Load previous conversations from Supabase
    try:
        recent_messages = supabase_memory.get_recent_messages(
            user_id, session_id, limit=20
        )
        
        # Convert to LangChain format
        chat_history = []
        for msg in recent_messages:
            if msg["role"] == "user":
                chat_history.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "assistant", "content": msg["content"]})
        
        if recent_messages:
            print(f"✓ Loaded {len(recent_messages)} previous messages from Supabase")
    except Exception as e:
        print(f"Warning: Could not load previous conversations: {e}")
        chat_history = []
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.2,
        max_output_tokens=1000,
    )
    
    # Use LangChain's built-in tools
    tools = [
        DuckDuckGoSearchRun(),      # Web search via DuckDuckGo
        SleepTool(),               # Time delay tool
    ]
    
    # Standard ReAct prompt
    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant with access to tools. You can use tools to help answer questions.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""
    )
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create standard memory with pre-loaded chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )
    
    # Pre-populate memory with chat history from Supabase
    for msg in chat_history:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    
    return agent_executor, supabase_memory, user_id, session_id

def main():
    """Main function to run the ReAct agent with Supabase memory."""
    
    # Check for API keys
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
        return
    
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_ANON_KEY"):
        print("Error: Supabase credentials not set.")
        print("Please set:")
        print("export SUPABASE_URL='your-supabase-url'")
        print("export SUPABASE_ANON_KEY='your-supabase-anon-key'")
        return
    
    # Get user and session info
    user_id = os.environ.get("CHAT_USER_ID", "default_user")
    session_id = os.environ.get("CHAT_SESSION_ID", str(uuid.uuid4()))
    
    print("LangChain ReAct Agent (with Supabase Memory)")
    print("=" * 50)
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print("Using LangChain's built-in tools: DuckDuckGo search, Sleep")
    print("Conversations are stored in Supabase cloud database")
    print("Type 'exit' to quit, 'sessions' to see your sessions, 'users' to see all users.")
    
    # Create agent
    try:
        agent_executor, supabase_memory, user_id, session_id = create_supabase_react_agent(user_id, session_id)
        if agent_executor is None:
            return
        print("✓ Agent created successfully!")
        
        # Show existing sessions for this user
        sessions = supabase_memory.get_user_sessions(user_id)
        if sessions:
            print(f"Found {len(sessions)} existing session(s) for user {user_id}")
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return
    
    # One-shot mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            # Save user input to Supabase
            supabase_memory.add_message(user_id, session_id, "user", query)
            
            result = agent_executor.invoke({"input": query})
            print(result["output"])
            
            # Save bot response to Supabase
            supabase_memory.add_message(user_id, session_id, "assistant", result["output"])
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Interactive mode
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye!")
                break
            elif user_input.strip().lower() == "sessions":
                sessions = supabase_memory.get_user_sessions(user_id)
                if sessions:
                    print(f"Your sessions:")
                    for session_id, timestamp in sessions.items():
                        print(f"  - {session_id} (last activity: {timestamp})")
                else:
                    print("No sessions found.")
                continue
            elif user_input.strip().lower() == "users":
                users = supabase_memory.get_all_users()
                print(f"All users: {', '.join(users)}")
                continue
            
            try:
                # Save user input to Supabase
                supabase_memory.add_message(user_id, session_id, "user", user_input)
                
                result = agent_executor.invoke({"input": user_input})
                print(f"Bot: {result['output']}")
                
                # Save bot response to Supabase
                supabase_memory.add_message(user_id, session_id, "assistant", result['output'])
            except Exception as e:
                print(f"Bot: Error processing your request: {e}")
                
    except KeyboardInterrupt:
        print("\nBot: Goodbye!")

if __name__ == "__main__":
    main()
