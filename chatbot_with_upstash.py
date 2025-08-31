from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, SleepTool
from langchain.tools import Tool

try:
    from upstash_redis import Redis
except ImportError:
    print("Error: upstash-redis not installed. Install with: pip install upstash-redis")
    sys.exit(1)

class UpstashMemory:
    """Upstash Redis-based conversation memory for scalable storage."""
    
    def __init__(self, upstash_url: str, upstash_token: str):
        self.redis = Redis(url=upstash_url, token=upstash_token)
        print("✓ Connected to Upstash Redis")
    
    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        try:
            message_key = f"chat:{user_id}:{session_id}:{int(time.time() * 1000)}"
            # Set each field individually for Upstash Redis compatibility
            self.redis.hset(message_key, "role", role)
            self.redis.hset(message_key, "content", content)
            self.redis.hset(message_key, "timestamp", str(time.time()))
            # Set expiration to 30 days
            self.redis.expire(message_key, 30 * 24 * 60 * 60)
            return True
        except Exception as e:
            print(f"Error adding message to Upstash: {e}")
            return False
    
    def get_recent_messages(self, user_id: str, session_id: str, limit: int = 10):
        """Get recent messages for a user session."""
        try:
            # Get all message keys for this user and session
            pattern = f"chat:{user_id}:{session_id}:*"
            keys = self.redis.keys(pattern)
            
            if not keys:
                return []
            
            # Sort keys by timestamp (extracted from key)
            keys.sort(key=lambda x: int(x.split(':')[-1]))
            
            # Get the most recent messages
            recent_keys = keys[-limit:] if len(keys) > limit else keys
            
            messages = []
            for key in recent_keys:
                message_data = self.redis.hgetall(key)
                if message_data:
                    messages.append({
                        "role": message_data.get('role', ''),
                        "content": message_data.get('content', '')
                    })
            
            return messages
        except Exception as e:
            print(f"Error getting messages from Upstash: {e}")
            return []
    
    def get_user_sessions(self, user_id: str):
        """Get all sessions for a user."""
        try:
            pattern = f"chat:{user_id}:*"
            keys = self.redis.keys(pattern)
            
            sessions = set()
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 3:
                    sessions.add(parts[2])
            
            return list(sessions)
        except Exception as e:
            print(f"Error getting user sessions from Upstash: {e}")
            return []
    
    def get_all_users(self):
        """Get all users who have conversations."""
        try:
            pattern = "chat:*"
            keys = self.redis.keys(pattern)
            
            users = set()
            for key in keys:
                parts = key.split(':')
                if len(parts) >= 2:
                    users.add(parts[1])
            
            return list(users)
        except Exception as e:
            print(f"Error getting users from Upstash: {e}")
            return []

# Custom Tools
def get_current_time(query: str = "") -> str:
    """Get the current date and time."""
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_math(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations (+, -, *, /, parentheses) are allowed."
        
        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def create_react_agent_with_memory(user_id: str, session_id: str, upstash_memory: UpstashMemory):
    """Create a ReAct agent with Upstash memory using LangGraph's built-in create_react_agent."""
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.1,
        max_output_tokens=1500,
    )
    
    # Create custom tools
    current_time_tool = Tool(
        name="get_current_time",
        description="Get the current date and time. Use this when someone asks about today's date, current time, or what day it is.",
        func=get_current_time
    )
    
    math_tool = Tool(
        name="calculate_math",
        description="Calculate mathematical expressions. Use this for any math calculations, arithmetic, or numerical problems.",
        func=calculate_math
    )
    
    # All available tools
    tools = [
        DuckDuckGoSearchRun(),      # Web search via DuckDuckGo
        current_time_tool,          # Custom time tool
        math_tool,                  # Custom math tool
        SleepTool(),               # Time delay tool
    ]
    
    # Enhanced prompt with context awareness
    enhanced_prompt = """You are a helpful AI assistant with access to powerful tools and conversation memory. You should actively use these tools to provide accurate and helpful information.

IMPORTANT: You have access to these tools and should use them when appropriate:
- Use DuckDuckGo search for current information, news, facts, or anything you're not certain about
- Use get_current_time for date/time questions
- Use calculate_math for any mathematical calculations
- Use SleepTool if you need to pause

You also have access to conversation history, so you can reference previous messages and maintain context.

Remember: If you're not certain about something or need current information, use the search tool. If someone asks about time/date, use the time tool. If there's math involved, use the math tool.

Be helpful, informative, and use your tools to provide the best possible answers while maintaining conversation context."""

    # Create ReAct agent using LangGraph's built-in function
    agent_graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=enhanced_prompt,
        version="v2"
    )
    
    return agent_graph

def extract_response(result):
    """Extract the final response from LangGraph result."""
    response_messages = result.get("messages", [])
    
    if response_messages:
        # Find the last AIMessage (final response)
        for msg in reversed(response_messages):
            # Handle AIMessage objects
            if hasattr(msg, 'content') and hasattr(msg, 'response_metadata'):
                response = msg.content
                if response:  # Only use if content is not empty
                    return response
            # Handle dict format
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                response = msg.get("content", "")
                if response:
                    return response
    
    return None

def main():
    """Main function to run the ReAct agent with Upstash memory."""
    
    # Check for required environment variables
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
        return
    
    if not os.environ.get("UPSTASH_REDIS_URL") or not os.environ.get("UPSTASH_REDIS_TOKEN"):
        print("Error: Upstash Redis credentials not set.")
        print("Please set them with:")
        print("export UPSTASH_REDIS_URL='your-upstash-redis-url'")
        print("export UPSTASH_REDIS_TOKEN='your-upstash-redis-token'")
        return
    
    print("🤖 LangGraph ReAct Agent with Upstash Memory")
    print("=" * 50)
    print("✓ Using LangGraph's built-in create_react_agent")
    print("✓ Tools: Web search, Time, Math, Sleep")
    print("✓ Database: Upstash Redis for conversation context")
    print("✓ Type 'exit' to quit, 'sessions' to see history")
    
    # Initialize Upstash memory
    upstash_memory = UpstashMemory(
        upstash_url=os.environ.get("UPSTASH_REDIS_URL"),
        upstash_token=os.environ.get("UPSTASH_REDIS_TOKEN")
    )
    
    # Generate user and session IDs
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    
    # Create agent
    try:
        agent_graph = create_react_agent_with_memory(user_id, session_id, upstash_memory)
        print("✓ Agent created successfully!")
    except Exception as e:
        print(f"Error creating agent: {e}")
        return
    
    # One-shot mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            # Save user input to Upstash
            upstash_memory.add_message(user_id, session_id, "user", query)
            
            # Get conversation context
            recent_messages = upstash_memory.get_recent_messages(user_id, session_id, limit=5)
            
            # Create messages array with context
            messages = [{"role": "user", "content": query}]
            if recent_messages:
                # Add recent context (excluding the current message)
                for msg in recent_messages[:-1]:  # Exclude the last message as it's the current one
                    messages.insert(0, msg)
            
            # Invoke agent
            inputs = {"messages": messages}
            result = agent_graph.invoke(inputs)
            
            # Extract and display response
            response = extract_response(result)
            if response:
                print(f"Bot: {response}")
                # Save bot response to Upstash
                upstash_memory.add_message(user_id, session_id, "assistant", response)
            else:
                print("Bot: No response generated")
                
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Interactive mode
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye! Your conversation has been saved to Upstash.")
                break
            
            if user_input.strip().lower() == "sessions":
                sessions = upstash_memory.get_user_sessions(user_id)
                print(f"Your sessions: {sessions}")
                continue
            
            try:
                # Save user input to Upstash
                upstash_memory.add_message(user_id, session_id, "user", user_input)
                
                # Get conversation context
                recent_messages = upstash_memory.get_recent_messages(user_id, session_id, limit=5)
                
                # Create messages array with context
                messages = [{"role": "user", "content": user_input}]
                if recent_messages:
                    # Add recent context (excluding the current message)
                    for msg in recent_messages[:-1]:  # Exclude the last message as it's the current one
                        messages.insert(0, msg)
                
                # Invoke agent
                inputs = {"messages": messages}
                result = agent_graph.invoke(inputs)
                
                # Extract and display response
                response = extract_response(result)
                if response:
                    print(f"Bot: {response}")
                    # Save bot response to Upstash
                    upstash_memory.add_message(user_id, session_id, "assistant", response)
                else:
                    print("Bot: No response generated")
                    
            except Exception as e:
                print(f"Bot: Error processing your request: {e}")
                
    except KeyboardInterrupt:
        print("\nBot: Goodbye! Your conversation has been saved to Upstash.")

if __name__ == "__main__":
    main()
