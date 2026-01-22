import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import JSONNodeParser

# --- КОНФИГУРАЦИЯ ---
PERSIST_DIR = "./index_store"
DATA_DIR = "./data"

# 1. Настройка моделей (Ollama)
# Мы задаем их глобально через Settings, чтобы LlamaIndex использовал их везде
print("⚙️ Инициализация моделей Ollama...")

# Генеративная модель (Qwen3 8B)
Settings.llm = Ollama(
    model="qwen3", 
    base_url="http://localhost:11434",
    request_timeout=300.0, # Локальная модель может думать долго
    temperature=0          # Для RAG нужна точность, а не креатив
)

# Эмбеддинг модель (Nomic)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)

def get_index():
    """
    Создает или загружает векторный индекс.
    Pattern: Checkpointer (Персистенция)
    """
    if not os.path.exists(PERSIST_DIR):
        print(f"📂 Индекс не найден в {PERSIST_DIR}. Создаем новый...")
        
        # Читаем JSON (LlamaIndex умный, он сам распарсит структуру)
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        print(f"📄 Загружено документов: {len(documents)}")
        
        # Создаем индекс (Тут идет векторизация через nomic-embed-text)
        index = VectorStoreIndex.from_documents(documents)
        
        # Сохраняем на диск
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("💾 Индекс сохранен!")
    else:
        print(f"🚀 Загружаем существующий индекс из {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        
    return index

def get_rag_tool_function():
    """
    Возвращает функцию поиска, которую можно скормить Агенту.
    """
    index = get_index()
    # Создаем движок запросов (top_k=3 - берем 3 самых похожих куска)
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    def search_knowledge_base(query: str) -> str:
        """Поиск информации в базе знаний технической поддержки."""
        response = query_engine.query(query)
        # Возвращаем текст ответа + источники (метаданные)
        return str(response)
    
    return search_knowledge_base

# Блок для быстрого теста (если запускаем файл напрямую)
if __name__ == "__main__":
    tool = get_rag_tool_function()
    print("\n--- ТЕСТ ПОИСКА ---")
    res = tool("Как исправить ошибку 429?")
    print(res)