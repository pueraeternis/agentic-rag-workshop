import sys
from typing import Literal

# Импорты
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage  # <--- Новый импорт
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Наш RAG
from rag_engine import get_rag_tool_function

# --- НАСТРОЙКА ---

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
model = ChatOllama(
    model="qwen3",
    base_url="http://localhost:11434",
    temperature=0,
)

# 3. Память
memory = MemorySaver()

# 4. Агент (Минималистичный вызов)
agent_executor = create_agent(
    model=model, 
    tools=tools, 
    checkpointer=memory
)

# --- ИНТЕРФЕЙС ---

def main():
    print("🤖 Ассистент готов к работе! (Введите 'q' для выхода)")
    
    # ID сессии
    config = {"configurable": {"thread_id": "session_1"}}
    
    # Системное сообщение (Роль)
    sys_msg = SystemMessage(content="Ты — ассистент техподдержки. Ищи ответы в базе знаний через lookup_policy. Отвечай на русском.")

    while True:
        try:
            user_input = input("\nВы: ")
            if user_input.lower() in ["q", "exit", "quit"]:
                print("До свидания!")
                break
            
            print("⏳ Агент думает...", end="", flush=True)
            
            last_message = ""
            # Передаем роль каждый раз в контексте (LangGraph разберется)
            for event in agent_executor.stream(
                {"messages": [sys_msg, ("user", user_input)]}, 
                config=config
            ):
                if "agent" in event:
                    print(".", end="", flush=True)
                if "tools" in event:
                    print(" [Поиск в базе] ", end="", flush=True)

            snapshot = agent_executor.get_state(config)
            if snapshot.values["messages"]:
                last_message = snapshot.values["messages"][-1].content
                print(f"\n\n🤖 Ассистент:\n{last_message}")
                
        except KeyboardInterrupt:
            print("\nВыход...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()