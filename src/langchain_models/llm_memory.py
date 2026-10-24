from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 👉 Initialized with the correct, latest Gemini model variant
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    
    # Check for exit before adding it to history
    if user_input.strip().lower() == 'exit':
        break
        
    chat_history.append(HumanMessage(content=user_input))
    
    # Pass the history list directly to Gemini
    result = model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print("\n--- Final Chat History Log ---")
print(chat_history)
