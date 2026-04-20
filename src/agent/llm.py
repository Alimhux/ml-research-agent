import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
load_dotenv()   


def get_llm(model_name: str = "yandexgpt-5.1/latest", temperature: float = 0) -> ChatOpenAI: 
    folder_id = os.getenv("YANDEX_CLOUD_FOLDER")
    api_key = os.getenv("YANDEX_CLOUD_API_KEY")

    return ChatOpenAI(
        model=f"gpt://{folder_id}/{model_name}",
        temperature=temperature,    
        api_key=api_key,
        base_url="https://ai.api.cloud.yandex.net/v1"
    )


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="Ты помощник по ML research."),
        HumanMessage(content="Что такое GRPO одним предложением?")
    ])
    print(response.content)