import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Initialize the Gemini LLM
# (Make sure GOOGLE_API_KEY is available in your system environment)
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)

# 2. Design the prompt template with history tracking
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, witty AI assistant powered by Gemini."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 3. Chain components using LangChain Expression Language (LCEL)
chain = prompt | model

# 4. Initialize session storage for chat history
sessions_db = {}

def get_session_history(session_id: str):
    if session_id not in sessions_db:
        sessions_db[session_id] = InMemoryChatMessageHistory()
    return sessions_db[session_id]

# 5. Wrap the chain with the message history manager
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Execute Multi-Turn Chat Simulation ---
config = {"configurable": {"session_id": "user_42"}}

# Turn 1
response1 = conversational_chain.invoke({"input": "Hi! My name is Alex and I love coding in Python."}, config)
print(f"AI: {response1.content}")

# Turn 2
response2 = conversational_chain.invoke({"input": "What is my name and favorite programming language?"}, config)
print(f"AI: {response2.content}")
