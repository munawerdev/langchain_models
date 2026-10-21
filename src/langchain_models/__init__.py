from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    result = model.invoke("what is the capital of pakistan")
    print(result.content)