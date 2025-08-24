from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define chatbot state
class ChatState(TypedDict):
    step: str
    name: str
    issue: str
    message: str

# Define node functions
def start_node(state: ChatState):
    return {"step": "GREETING", "message": "Hello! I'm your assistant. What's your name?"}

def greeting_node(state: ChatState):
    return {
        "step": "ASK_ISSUE",
        "name": state["message"],  # user input stored in "message"
        "message": f"Nice to meet you {state['message']}! What can I help you with?",
    }

def ask_issue_node(state: ChatState):
    return {
        "step": "RESOLVE",
        "issue": state["message"],
        "message": f"Thanks for sharing. I'll note your issue: '{state['message']}'. Do you want to end the chat?",
    }

def resolve_node(state: ChatState):
    user_input = state["message"].lower()
    if user_input in ["yes", "y", "bye"]:
        return {"step": "END", "message": f"Goodbye {state.get('name', '')}, have a great day!"}
    else:
        return {"step": "ASK_ISSUE", "message": "Okay, please tell me more about your issue."}

# Build workflow graph
graph = StateGraph(ChatState)

graph.add_node("START", start_node)
graph.add_node("GREETING", greeting_node)
graph.add_node("ASK_ISSUE", ask_issue_node)
graph.add_node("RESOLVE", resolve_node)

# Define edges - simple linear flow
graph.add_edge("START", "GREETING")
graph.add_edge("GREETING", "ASK_ISSUE")
graph.add_edge("ASK_ISSUE", "RESOLVE")
graph.add_edge("RESOLVE", END)

# Set entry point
graph.set_entry_point("START")

# Compile the graph
app = graph.compile()

# --- Running the chatbot ---
print("Starting LangGraph Chatbot...")
print("=" * 40)

# Initialize state with empty message
state = {"step": "START", "name": "", "issue": "", "message": ""}

# Get the first bot message
state = app.invoke(state)
print("Bot:", state["message"])

# Main chat loop - step by step
try:
    # Step 1: Get user's name
    user_in = input("You: ")
    state["message"] = user_in
    state = app.invoke(state)
    print("Bot:", state["message"])
    
    # Step 2: Get user's issue
    user_in = input("You: ")
    state["message"] = user_in
    state = app.invoke(state)
    print("Bot:", state["message"])
    
    # Step 3: Ask if they want to end
    user_in = input("You: ")
    state["message"] = user_in
    state = app.invoke(state)
    print("Bot:", state["message"])
    
except KeyboardInterrupt:
    print("\nBot: Goodbye! Thanks for chatting!")

print("=" * 40)
print("Chat ended.")
