import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["YANDEX_CLOUD_API_KEY"] = "AQVNySLQcZQ5L_3HhUJvMwo-421kxWid2Wl2H2Eo"
os.environ["YANDEX_CLOUD_FOLDER"] = "b1gc7ekb4govoh9ie7r8"


def get_llm(model_name: str = "yandexgpt-5.1/latest", temperature: float = 0) -> ChatOpenAI: 
    folder_id = os.environ["YANDEX_CLOUD_FOLDER"]
    api_key = os.environ["YANDEX_CLOUD_API_KEY"]

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