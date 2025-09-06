

import os
import sys
import time
import uuid
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, SleepTool
from langchain.tools import Tool

try:
    from upstash_redis import Redis
except ImportError:
    print("Error: upstash-redis not installed. Install with: pip install upstash-redis")
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: scikit-learn not installed. Install with: pip install scikit-learn")
    sys.exit(1)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    source_path: str
    chunk_text: str
    chunk_index: int


class TfIdfRagIndex:
    """TF-IDF based document retrieval system."""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._chunks = chunks
        corpus: List[str] = [c.chunk_text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)
        print(f"✓ RAG Index built with {len(chunks)} document chunks")

    def search(self, query: str, k: int = 3) -> List[Tuple[float, DocumentChunk]]:
        """Search for relevant document chunks."""
        if not query.strip():
            return []
        
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._matrix)[0]
        
        scored: List[Tuple[float, DocumentChunk]] = [
            (float(similarities[i]), self._chunks[i]) 
            for i in range(len(self._chunks))
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:max(1, k)]


class UpstashMemory:
    """Upstash Redis-based conversation memory for scalable storage."""
    
    def __init__(self, upstash_url: str, upstash_token: str):
        self.redis = Redis(url=upstash_url, token=upstash_token)
        print("✓ Connected to Upstash Redis")
    
    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """Add a message to the conversation history."""
        try:
            message_key = f"chat:{user_id}:{session_id}:{int(time.time() * 1000)}"
            self.redis.hset(message_key, "role", role)
            self.redis.hset(message_key, "content", content)
            self.redis.hset(message_key, "timestamp", str(time.time()))
            self.redis.expire(message_key, 30 * 24 * 60 * 60)  # 30 days
            return True
        except Exception as e:
            print(f"Error adding message to Upstash: {e}")
            return False
    
    def get_recent_messages(self, user_id: str, session_id: str, limit: int = 10):
        """Get recent messages for a user session."""
        try:
            pattern = f"chat:{user_id}:{session_id}:*"
            keys = self.redis.keys(pattern)
            
            if not keys:
                return []
            
            keys.sort(key=lambda x: int(x.split(':')[-1]))
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


def read_text_files_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    """Read all .txt files from a directory."""
    if not os.path.isdir(directory_path):
        return []
    
    texts: List[Tuple[str, str]] = []
    for file_path in sorted(glob.glob(os.path.join(directory_path, "**", "*.txt"), recursive=True)):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append((file_path, f.read()))
        except Exception:
            continue
    return texts


def split_text_into_chunks(text: str, max_characters: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if max_characters <= 0:
        return [text]
    
    chunks: List[str] = []
    start_index: int = 0
    text_length: int = len(text)
    stride: int = max(1, max_characters - overlap)
    
    while start_index < text_length:
        end_index: int = min(start_index + max_characters, text_length)
        chunk: str = text[start_index:end_index].strip()
        if chunk:
            chunks.append(chunk)
        start_index += stride
    
    return chunks


def build_rag_index(docs_directory: str) -> TfIdfRagIndex:
    """Build the RAG index from documents."""
    files = read_text_files_from_directory(docs_directory)
    if not files:
        print(f"Warning: No .txt files found in '{docs_directory}'.")
        print("RAG functionality will be limited. Add .txt files to enable document retrieval.")
        return TfIdfRagIndex([])

    chunks: List[DocumentChunk] = []
    for path, text in files:
        for i, chunk_text in enumerate(split_text_into_chunks(text)):
            chunks.append(DocumentChunk(
                source_path=path, 
                chunk_text=chunk_text, 
                chunk_index=i
            ))
    
    return TfIdfRagIndex(chunks)


# Custom Tools
def get_current_time(query: str = "") -> str:
    """Get the current date and time."""
    return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def calculate_math(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations (+, -, *, /, parentheses) are allowed."
        
        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def create_rag_search_tool(rag_index: TfIdfRagIndex) -> Tool:
    """Create a RAG search tool for the agent."""
    def rag_search(query: str) -> str:
        """Search documents for relevant information."""
        if not rag_index._chunks:
            return "No documents available for search. Add .txt files to the docs/ folder."
        
        results = rag_index.search(query, k=3)
        if not results:
            return "No relevant information found in the documents."
        
        response_parts = []
        for score, chunk in results:
            source_name = os.path.basename(chunk.source_path)
            response_parts.append(f"[Source: {source_name}]\n{chunk.chunk_text}\n")
        
        return "\n".join(response_parts)
    
    return Tool(
        name="search_documents",
        description="Search through uploaded documents for relevant information. Use this when the user asks about specific content, facts, or information that might be in the documents.",
        func=rag_search
    )


def create_react_agent_with_rag_and_memory(user_id: str, session_id: str, upstash_memory: UpstashMemory, rag_index: TfIdfRagIndex):
    """Create a ReAct agent with RAG, Upstash memory, and tools."""
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.1,
        max_output_tokens=2000,
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
    
    rag_search_tool = create_rag_search_tool(rag_index)
    
    # All available tools
    tools = [
        rag_search_tool,           # RAG document search
        DuckDuckGoSearchRun(),     # Web search via DuckDuckGo
        current_time_tool,         # Custom time tool
        math_tool,                 # Custom math tool
        SleepTool(),              # Time delay tool
    ]
    
    # Enhanced prompt with RAG awareness
    enhanced_prompt = """You are a helpful AI assistant with access to powerful tools, document search capabilities, and conversation memory.

CRITICAL: You MUST use the search_documents tool FIRST when users ask about any specific content, companies, or information that might be in your documents.

IMPORTANT: You have access to these tools and should use them when appropriate:
- ALWAYS use search_documents FIRST to find information from uploaded documents when users ask about specific content, companies, or facts
- Use DuckDuckGo search ONLY if document search doesn't find relevant information
- Use get_current_time for date/time questions
- Use calculate_math for any mathematical calculations
- Use SleepTool if you need to pause

You also have access to conversation history, so you can reference previous messages and maintain context.

RULE: When users ask questions about companies, products, or any specific information, you MUST search your documents first using the search_documents tool before providing any answer.

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
        for msg in reversed(response_messages):
            if hasattr(msg, 'content') and hasattr(msg, 'response_metadata'):
                response = msg.content
                if response:
                    return response
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                response = msg.get("content", "")
                if response:
                    return response
    
    return None


def main():
    """Main function to run the RAG chatbot with Upstash memory."""
    
    # Check for required environment variables
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
        print("Or create a .env file with your API keys (see env_template.txt)")
        return
    
    if not os.environ.get("UPSTASH_REDIS_URL") or not os.environ.get("UPSTASH_REDIS_TOKEN"):
        print("Error: Upstash Redis credentials not set.")
        print("Please set them with:")
        print("export UPSTASH_REDIS_URL='your-upstash-redis-url'")
        print("export UPSTASH_REDIS_TOKEN='your-upstash-redis-token'")
        print("Or create a .env file with your credentials (see env_template.txt)")
        return
    
    print("🤖 RAG Chatbot with Upstash Memory")
    print("=" * 50)
    print("✓ Using LangGraph's built-in create_react_agent")
    print("✓ Tools: Document search (RAG), Web search, Time, Math, Sleep")
    print("✓ Database: Upstash Redis for conversation context")
    print("✓ Type 'exit' to quit, 'sessions' to see history")
    
    # Initialize RAG index
    docs_dir = os.environ.get("RAG_DOCS_DIR", "docs")
    rag_index = build_rag_index(docs_dir)
    
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
        agent_graph = create_react_agent_with_rag_and_memory(user_id, session_id, upstash_memory, rag_index)
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
                for msg in recent_messages[:-1]:
                    messages.insert(0, msg)
            
            # Invoke agent
            inputs = {"messages": messages}
            result = agent_graph.invoke(inputs)
            
            # Extract and display response
            response = extract_response(result)
            if response:
                print(f"Bot: {response}")
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
                    for msg in recent_messages[:-1]:
                        messages.insert(0, msg)
                
                # Invoke agent
                inputs = {"messages": messages}
                result = agent_graph.invoke(inputs)
                
                # Extract and display response
                response = extract_response(result)
                if response:
                    print(f"Bot: {response}")
                    upstash_memory.add_message(user_id, session_id, "assistant", response)
                else:
                    print("Bot: No response generated")
                    
            except Exception as e:
                print(f"Bot: Error processing your request: {e}")
                
    except KeyboardInterrupt:
        print("\nBot: Goodbye! Your conversation has been saved to Upstash.")


if __name__ == "__main__":
    main()
