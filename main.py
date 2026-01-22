from typing import TYPE_CHECKING

# --- ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ---
from dotenv import load_dotenv

# Импорты LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

# Импорты LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Наш RAG
from rag_engine import get_rag_tool_function

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig

# --- НАСТРОЙКА ---
load_dotenv()

# 1. Инструмент
rag_search_func = get_rag_tool_function()


@tool
def lookup_policy(query: str) -> str:
    """
    Используй этот инструмент, чтобы найти информацию о технических проблемах,
    настройках SSO, оплате, API или багах в базе знаний компании.
    Вход: конкретный поисковый запрос.
    """
    return rag_search_func(query)


tools = [lookup_policy]

# 2. Модель
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# 3. Память
memory = MemorySaver()


# --- ГРАФ ---


def call_model(state: MessagesState):
    """Узел агента"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=memory)


# --- ИНТЕРФЕЙС ---


def main():
    print("🤖 Ассистент готов к работе! (Введите 'q' для выхода)")

    # 1. Инициализируем хендлер
    try:
        langfuse_handler = CallbackHandler()
        print("✅ Langfuse мониторинг подключен")
    except Exception as e:
        print(f"⚠️ Ошибка подключения Langfuse: {e}")
        langfuse_handler = None

    # 2. Добавляем его в конфиг
    # ИСПРАВЛЕНИЕ: Явно аннотируем тип списка как List[BaseCallbackHandler].
    # Это удовлетворяет инвариантность списков (List[Parent] принимает Child).
    callbacks: list[BaseCallbackHandler] = [langfuse_handler] if langfuse_handler else []

    config: RunnableConfig = {
        "configurable": {"thread_id": "session_1"},
        "callbacks": callbacks,
    }

    sys_msg = SystemMessage(
        content="Ты — ассистент техподдержки. Ищи ответы в базе знаний через lookup_policy. Отвечай на русском.",
    )

    while True:
        try:
            user_input = input("\nВы: ")
            if user_input.lower() in ["q", "exit", "quit"]:
                print("До свидания!")
                break

            print("⏳ Агент думает...", end="", flush=True)

            inputs: MessagesState = {
                "messages": [sys_msg, HumanMessage(content=user_input)],
            }

            for event in app.stream(inputs, config=config):
                if "agent" in event:
                    print(".", end="", flush=True)
                if "tools" in event:
                    print(" [Поиск в базе] ", end="", flush=True)

            snapshot = app.get_state(config)
            if snapshot.values["messages"]:
                last_message = snapshot.values["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"\n\n🤖 Ассистент:\n{last_message.content}")

        except KeyboardInterrupt:
            print("\nВыход...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
