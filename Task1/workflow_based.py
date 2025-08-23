class WorkflowChatbot:
    def __init__(self):
        self.state = "START"
        self.user_data = {}

    def handle_input(self, user_input):
        if self.state == "START":
            self.state = "GREETING"
            return "Hello! I’m your assistant. What’s your name?"

        elif self.state == "GREETING":
            self.user_data["name"] = user_input
            self.state = "ASK_ISSUE"
            return f"Nice to meet you {user_input}!!! What can I help you with?"

        elif self.state == "ASK_ISSUE":
            self.user_data["issue"] = user_input
            self.state = "RESOLVE"
            return f"Thanks for sharing. I’ll note your issue: '{user_input}'. Do you want to end the chat?"

        elif self.state == "RESOLVE":
            if user_input.lower() in ["yes", "y", "bye"]:
                self.state = "END"
                return f"Goodbye {self.user_data.get('name', '')}, have a great day!"
            else:
                self.state = "ASK_ISSUE"
                return "Okay, please tell me more about your issue."

        elif self.state == "END":
            return "Chat has already ended."


if __name__ == "__main__":
    bot = WorkflowChatbot()
    print("Bot:", bot.handle_input(""))   # Kickstart
    
    while bot.state != "END":
        user_in = input("You: ")
        print("Bot:", bot.handle_input(user_in))
