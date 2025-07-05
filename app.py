import streamlit as st
import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import tempfile
import shutil # For removing directory


# --- Helper Function to Check Ollama Service ---
def is_ollama_running():
    try:
        ollama.list()
        return True
    except Exception:
        return False

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="本地智慧助手",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 您的本地智慧助手")
st.markdown("""
一个完全离线的助手，利用 **Gemma 3n 模型**在您的本地设备上运行。
输入您的任何问题或需求，它将尽力为您提供帮助。
""")

st.divider() # Adds a visual separator

# --- Sidebar for Settings ---
st.sidebar.title("设置")

# Check Ollama service status early
if not is_ollama_running():
    st.error("❗ Ollama 服务未运行。请在命令行中输入 `ollama serve` 启动服务，否则应用将无法工作。")
    st.stop() # Stop app execution if Ollama is not running

# Model selection
available_models = ["gemma3n"] # Default model
try:
    ollama_models = ollama.list()['models']
    for model_info in ollama_models:
        if model_info['model'] not in available_models:
            available_models.append(model_info['model'])
except Exception:
    st.sidebar.warning("无法获取 Ollama 模型列表。请确保 Ollama 正在运行。")

selected_model = st.sidebar.selectbox("选择模型", available_models)

# Validate if selected model is available
if selected_model not in [m['model'] for m in ollama.list()['models']]:
    st.warning(f"❗ 您选择的模型 `{selected_model}` 尚未下载。请在命令行中运行 `ollama pull {selected_model}`。")


# Temperature slider for model response creativity
temperature = st.sidebar.slider("生成温度 (Temperature)", 0.0, 2.0, 0.7, 0.05,
                                help="较高的值会使输出更随机，较低的值会使输出更集中和确定。")

# Clear chat history button
if st.sidebar.button("清空聊天记录"):
    st.session_state.messages = []
    st.rerun() # Rerun the app to clear displayed messages

# --- Dynamic System Instruction Mode ---
st.sidebar.markdown("---\n**助手模式**")
scenario_modes = {
    "通用助手": "你是一个乐于助人的助手。请根据提供的上下文信息（如果提供）和对话历史来回答问题。",
    "农业专家": "你是一个经验丰富的农业专家，请根据提供的农作物知识和最新农业技术，为用户提供专业的种植、病虫害防治和产量优化建议。你的回答应实用且易于理解。",
    "基础医疗咨询": "你是一个基础医疗信息助手。请根据提供的医学知识，为用户提供常见的健康问题、疾病预防和基础急救措施信息。请务必强调：你不能替代专业的医生诊断和治疗，所有信息仅供参考，如有健康问题请及时就医。",
    "天气灾害预警": "你是一个天气和灾害预警助手。请根据提供的气象数据和灾害应对知识，为用户提供天气预报、自然灾害（如洪水、地震、山体滑坡）的预警信息和紧急应对措施建议。强调及时关注官方预警和撤离通知。",
    "基础教育知识": "你是一个基础教育知识普及者。请用简单、清晰的语言解释各种基础科学、历史、地理等知识，帮助用户学习和理解基础概念。"
}
selected_mode = st.sidebar.selectbox("选择助手模式", list(scenario_modes.keys()))
system_instruction = scenario_modes[selected_mode]


st.sidebar.markdown("---\n**知识库 (RAG)**")

# --- RAG Specific Configuration and Functions ---
# Persistent directory for ChromaDB
# This will create a 'chroma_db_rag' folder next to app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db_rag")

if "vectorstore" not in st.session_state:
    st.session_state.chroma_db_dir = PERSIST_DIRECTORY
    if not os.path.exists(st.session_state.chroma_db_dir):
        os.makedirs(st.session_state.chroma_db_dir)
    
    # Initialize Ollama Embeddings
    try:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Check if nomic-embed-text is actually pulled
        if "nomic-embed-text" not in [m['model'] for m in ollama.list()['models']]:
            st.warning("❗ 嵌入模型 `nomic-embed-text` 尚未下载。请在命令行中运行 `ollama pull nomic-embed-text`。")
            st.session_state.embeddings = None # Disable RAG if embedder is not ready
        else:
            st.session_state.vectorstore = Chroma(
                embedding_function=st.session_state.embeddings,
                persist_directory=st.session_state.chroma_db_dir
            )
            st.sidebar.success("知识库加载成功！")
    except Exception as e:
        st.sidebar.error(f"知识库初始化失败: {e}. 请确保 Ollama 正在运行且 'nomic-embed-text' 模型已下载。")
        st.session_state.embeddings = None
        st.session_state.vectorstore = None


def process_documents(uploaded_files):
    if st.session_state.vectorstore is None:
        st.error("知识库未初始化或嵌入模型未准备好，请检查 Ollama 服务和嵌入模型。")
        return

    documents = []
    
    # Progress bar for file loading
    loading_progress_bar = st.sidebar.progress(0, text="正在加载文件...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == ".txt" or file_extension == ".md":
                loader = TextLoader(tmp_file_path, encoding="utf-8")
            else:
                st.warning(f"不支持的文件类型: {file_extension}。跳过 {uploaded_file.name}")
                continue
            
            docs = loader.load()
            if not docs or not any(d.page_content.strip() for d in docs): # Check if any content is extracted
                st.warning(f"文件 {uploaded_file.name} 未检测到任何文本内容或内容为空白。请确保是可选择文本的PDF或有内容的文本文件。")
            documents.extend(docs)
        except Exception as e:
            st.error(f"加载文件 {uploaded_file.name} 失败: {e}")
        finally:
            os.remove(tmp_file_path) # Clean up temp file
        
        loading_progress_bar.progress((i + 1) / len(uploaded_files), text=f"正在加载文件 {i+1}/{len(uploaded_files)}...")
    
    loading_progress_bar.empty() # Clear loading progress bar

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        if splits:
            # Progress bar for embedding and adding to vectorstore
            embedding_progress_bar = st.sidebar.progress(0, text="正在处理文档并构建知识库...")
            total_splits = len(splits)
            
            # Add documents in batches to show progress
            batch_size = 20 # Adjust batch size based on memory
            for i in range(0, total_splits, batch_size):
                batch_splits = splits[i:min(i + batch_size, total_splits)]
                try:
                    st.session_state.vectorstore.add_documents(batch_splits)
                except Exception as e:
                    st.error(f"将文档批次添加到知识库失败: {e}")
                    break # Stop if a batch fails
                
                progress = (i + len(batch_splits)) / total_splits
                embedding_progress_bar.progress(progress, text=f"正在处理文档并构建知识库: {int(progress*100)}%")
            
            st.session_state.vectorstore.persist() # Save changes to disk
            embedding_progress_bar.empty() # Clear embedding progress bar
            st.success(f"成功将 {len(splits)} 个文档块添加到知识库。")
        else:
            st.warning("未从文档中提取到任何有效文本块。")
    else:
        st.info("未处理任何文档。")


uploaded_files = st.sidebar.file_uploader(
    "上传您的知识文档 (TXT, MD, PDF)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
    help="请上传包含可选择文本的文档。扫描版PDF可能无法提取文本。"
)

if uploaded_files:
    if st.sidebar.button("处理上传文件", key="process_files_btn"):
        process_documents(uploaded_files)

# Clear ChromaDB persistent directory as well
if st.sidebar.button("清除知识库", key="clear_db_btn"):
    if st.session_state.vectorstore:
        try:
            st.session_state.vectorstore.delete_collection() # Deletes all data in the collection
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY) # Remove the directory itself
                os.makedirs(PERSIST_DIRECTORY) # Recreate empty directory for future use

            # Re-initialize the vectorstore
            if st.session_state.embeddings: # Only if embeddings are available
                st.session_state.vectorstore = Chroma(
                    embedding_function=st.session_state.embeddings,
                    persist_directory=st.session_state.chroma_db_dir
                )
            st.success("知识库已清空。")
        except Exception as e:
            st.error(f"清空知识库失败: {e}")
    else:
        st.warning("知识库未初始化或已为空。")


st.sidebar.markdown("---\n")
st.sidebar.info("💡 这是一个完全离线的应用。所有数据处理都在您的本地设备上完成，不会上传到任何服务器。")


# --- "About" / Help Section ---
with st.sidebar.expander("关于 & 帮助"):
    st.markdown("""
    **版本:** 1.0.0
    **作者:** [您的名字/团队名]
    **简介:** 这是一个完全离线的本地智慧助手，利用 Ollama 平台运行 Gemma 3n 大语言模型，旨在为无网络或弱网络环境的用户提供智能对话和知识检索服务。

    **🚀 快速开始：**
    1.  **安装 Ollama：** 访问 `ollama.com` 下载并安装适用于您操作系统的 Ollama 应用程序。
    2.  **启动 Ollama 服务：** 打开命令行，运行 `ollama serve`。Ollama 桌面应用通常会自动在后台运行。
    3.  **下载模型：** 在命令行中运行 `ollama pull gemma3n` (大语言模型) 和 `ollama pull nomic-embed-text` (知识库嵌入模型)。这些模型需要一次性下载，之后即可离线使用。
    4.  **运行本应用：** 找到本应用的启动程序 (例如：`本地智慧助手.exe`) 双击运行。
    5.  **开始对话！**

    **📚 知识库 (RAG) 使用：**
    *   点击"上传您的知识文档"上传您想让模型学习的本地文件 (目前支持 TXT, MD, PDF)。
    *   点击"处理上传文件"按钮，应用会将文档内容添加到本地知识库。
        *   **注意：** PDF 文件必须包含可选择的文本层，扫描版 PDF 可能无法提取内容。
    *   模型在回答问题时会优先从知识库中检索相关信息。
    *   "清除知识库"按钮会删除所有已上传文档的索引，但不会删除原始文件。

    **❓ 常见问题：**
    *   **应用无响应 / 模型不工作：** 请确保 Ollama 服务正在后台运行，并且您已下载了 `gemma3n` 和 `nomic-embed-text` 模型。
    *   **PDF 文件未检测到文本：** 请确认 PDF 是可选择文本的，而非图片扫描件。
    *   **应用卡顿/速度慢：** 模型推理速度受限于您电脑的CPU性能。选择较小的模型或降低"生成温度"可能有所帮助。

    **感谢您使用本地智慧助手！**
    """)


# --- Initialize Session State (to store conversation history) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("您有什么问题或想了解的？"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a loading spinner while waiting for response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("思考中... 请稍候片刻，这取决于您的CPU性能"):
            try:
                # --- RAG Logic: Retrieve context from documents ---
                context = ""
                if st.session_state.vectorstore and st.session_state.embeddings:
                    with st.spinner("正在知识库中检索相关信息..."):
                        # Retrieve top 4 most relevant chunks
                        docs = st.session_state.vectorstore.similarity_search(prompt, k=4)
                        context = "\n".join([doc.page_content for doc in docs])
                        if context:
                            st.info("已从知识库中检索到相关信息。")
                            # print(f"Retrieved Context:\n{context}") # For debugging

                # Prepare messages for Ollama API, including context and dynamic system instruction
                messages_for_ollama = [
                    {"role": "system", "content": system_instruction} # Use dynamic system instruction
                ]
                if context:
                    messages_for_ollama.append({"role": "system", "content": f"相关上下文信息:\n{context}"})
                
                # Add existing conversation history
                for m in st.session_state.messages:
                    # Exclude system messages from previous turns if we are re-injecting them
                    if m["role"] != "system":
                        messages_for_ollama.append({"role": m["role"], "content": m["content"]})
                
                # Ollama Chat Call
                stream = ollama.chat(
                    model=selected_model,
                    messages=messages_for_ollama,
                    stream=True,
                    options=dict(temperature=temperature)
                )

                for chunk in stream:
                    if 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"与模型交互时发生错误: {e}")
                st.warning("请确保 Ollama 服务正在运行 (在命令行中输入 `ollama serve`) 并且您已下载了您选择的模型 (`ollama pull your_model_name`) 和嵌入模型 (`ollama pull nomic-embed-text`)。")

st.divider() # Another visual separator