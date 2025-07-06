# 🧠 本地智慧助手：偏远地区离线大模型服务解决方案

![Project Banner](https://via.placeholder.com/1200x400/2a3d4f/ffffff?text=Offline+Gemma+3n+Assistant)
*（这里可以替换为您的应用截图或定制化横幅，让项目更生动！）*

## ✨ 项目简介

欢迎来到 **本地智慧助手**！这是一个专为解决偏远地区网络基础设施不足问题而设计的大语言模型应用。我们利用 Google 最新的 **Gemma 3n 模型**，并结合 **Ollama 离线服务**，打造了一个完全在本地运行、无需互联网连接的智能问答平台。

我们的目标是让知识触手可及，即使在网络不便的环境下，用户也能通过智能助手获取信息、学习新知，赋能当地社区发展。

**为何选择它？**
- **100% 离线运行**：所有计算都在本地完成，无惧网络中断。
- **本地知识库增强 (RAG)**：允许用户上传本地文档，模型能基于私有数据回答问题。
- **多场景助手模式**：内置多种专业模式，满足不同领域（如农业、医疗、教育）的需求。
- **用户友好界面**：基于 Streamlit 构建，操作直观，易于上手。

## 🌟 核心亮点与功能

*   **离线大模型对话**：
    *   深度整合 [Ollama](https://ollama.com/) 平台，本地运行 Gemma 3n 模型，提供快速、流畅的对话体验。
    *   用户可以与模型进行自然语言交互，获取各种问题的答案。
*   **检索增强生成 (RAG) 知识库**：
    *   **文档上传**：支持 `TXT`, `MD`, `PDF` 等格式的本地文档上传。
    *   **智能检索**：应用会将文档内容智能分块、向量化并存储在本地 [ChromaDB](https://www.trychroma.com/) 向量数据库中。
    *   **上下文增强**：用户提问时，系统会从知识库中检索最相关的文档片段，并作为额外上下文传递给 Gemma 3n 模型，确保回答的准确性和针对性。
    *   **持久化存储**：知识库数据在应用关闭后依然保留，无需重复上传和处理。
*   **动态助手模式**：
    *   侧边栏提供多种预设的助手模式，例如“农业专家”、“基础医疗咨询”、“天气灾害预警”、“基础教育知识”等。
    *   用户可根据需求切换模式，模型将自动调整其回答风格和侧重点，提供更专业的服务。
*   **实时用户体验优化**：
    *   **流式响应**：模型回答以打字机效果逐步显示，提供更自然的交互感。
    *   **文件处理进度条**：在上传和处理知识文档时，清晰显示进度，避免用户等待焦虑。
    *   **Ollama 服务与模型状态检查**：智能检测 Ollama 服务是否运行、所需模型是否下载，并提供直观的引导和错误提示。
*   **简洁直观的 UI**：
    *   基于 Streamlit 框架构建，界面布局清晰，操作便捷。
    *   支持调节模型生成温度、清空聊天记录等实用功能。

## 🛠️ 技术栈

*   **核心框架**：[Python](https://www.python.org/)
*   **Web 应用框架**：[Streamlit](https://streamlit.io/)
*   **本地大模型服务**：[Ollama](https://ollama.com/) (运行 Gemma 3n 和 nomic-embed-text)
*   **LLM 编排与 RAG**：[LangChain](https://www.langchain.com/) (用于文档加载、文本分割、嵌入和向量存储集成)
*   **向量数据库**：[ChromaDB](https://www.trychroma.com/) (本地嵌入式向量数据库)
*   **PDF 解析**：[PyPDF](https://pypdf.readthedocs.io/en/stable/)

## 🚀 快速开始

### 前提条件

在运行此项目之前，请确保您的系统满足以下要求：

1.  **Python 3.8+**：推荐使用 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 进行环境管理。
2.  **Git**：用于克隆项目仓库。
3.  **Ollama**：访问 [ollama.com](https://ollama.com/) 下载并安装适用于您操作系统的 Ollama 应用程序。

### 安装与运行

1.  **克隆项目仓库：**
    ```bash
    git clone https://github.com/seveNine97/offline-gemma-assistant.git # 请替换为您的实际仓库名
    cd offline-gemma-assistant
    ```

2.  **创建并激活虚拟环境 (推荐)：**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **安装 Python 依赖：**
    ```bash
    pip install -r requirements.txt # 如果您创建了requirements.txt
    # 或者手动安装：
    pip install streamlit ollama langchain langchain-community chromadb pypdf
    ```
    *（建议：在项目根目录运行 `pip freeze > requirements.txt` 生成 `requirements.txt` 文件，方便他人安装。）*

4.  **启动 Ollama 服务：**
    在您的命令行中运行以下命令。Ollama 桌面应用通常会自动在后台运行，如果已运行则无需重复执行。
    ```bash
    ollama serve
    ```

5.  **下载大语言模型和嵌入模型：**
    这两个模型需要一次性下载，之后即可离线使用。
    ```bash
    ollama pull gemma3n
    ollama pull nomic-embed-text
    ```

6.  **运行本地智慧助手应用：**
    在项目根目录（`app.py` 所在目录）下运行：
    ```bash
    streamlit run app.py
    ```
    您的默认网页浏览器将自动打开一个新标签页，显示“本地智慧助手”应用程序。

## 💡 使用指南

*   **对话交流**：在底部的输入框中输入您的问题，点击“发送”或按回车键，模型将为您提供回答。
*   **选择助手模式**：在左侧的侧边栏中，您可以选择不同的“助手模式”（如农业专家、医疗咨询等），以获得针对特定领域的专业回答。
*   **知识库 (RAG)**：
    *   点击侧边栏中的“上传您的知识文档 (TXT, MD, PDF)”，选择您希望模型学习的本地文档。
    *   上传后，点击“处理上传文件”按钮。系统将处理文档并构建本地知识库。
    *   **注意**：PDF 文件必须是包含可选择文本层的，扫描版图片PDF可能无法提取文本。
    *   知识库构建完成后，模型在回答相关问题时将优先参考这些文档。
    *   点击“清除知识库”按钮可以清空所有已上传文档的索引。
*   **清空聊天记录**：点击侧边栏中的“清空聊天记录”按钮，可以开始新的对话。
*   **调节生成温度**：通过侧边栏的滑块调节“生成温度”，控制模型回答的随机性或确定性。

## 📂 项目结构

```bash
kaggle_competition/
├── app.py # Streamlit 主应用程序代码
├── requirements.txt # Python 依赖列表 (建议创建)
├── README.md # 项目说明文件 (当前文件)
├── chroma_db_rag/ # 持久化存储RAG知识库数据的文件夹 (自动生成)
├── venv/ # Python 虚拟环境 (本地创建)
├── .gitignore # Git 忽略文件 
```


## 📈 未来改进方向 (Kaggle 比赛额外加分项)

*   **更强大的打包方案**：探索将 Ollama 服务和模型文件一并打包到最终可执行文件中的可能性（例如通过自定义安装脚本），实现真正的“一键安装”。
*   **更多模型支持**：集成更多 Gemma 系列模型或其他适合本地运行的开源模型，并提供便捷的切换功能。
*   **高级 RAG 功能**：
    *   支持更多文档类型（如图片、表格）。
    *   实现更复杂的检索策略（例如：多查询检索、混合检索）。
    *   可视化知识库内容，方便用户管理。
*   **UI/UX 优化**：
    *   更美观的 Streamlit 主题或自定义 CSS。
    *   聊天消息的复制功能。
    *   性能监控和优化提示。
*   **离线微调 (LORA)**：虽然复杂，但若能实现轻量级离线微调（如基于 LORA 适配器），将模型适应特定用户数据，将是重大突破。
*   **多语言支持**：如果目标用户群体涉及多种语言，添加多语言界面支持。

## 🤝 贡献

我们欢迎所有对离线 AI 和偏远地区技术赋能感兴趣的开发者加入！如果您有任何想法或建议，欢迎提交 Pull Request 或创建 Issue。

## 📜 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 🙏 致谢

*   [Kaggle](https://www.kaggle.com/) 和 [Google](https://about.google/) 举办的 Gemma 3n 黑客马拉松。
*   [Ollama](https://ollama.com/) 团队为本地大模型服务做出的杰出贡献。
*   [Streamlit](https://streamlit.io/) 团队提供的优秀 Web 应用框架。
*   [LangChain](https://www.langchain.com/) 团队提供的强大 LLM 应用开发工具。
*   [ChromaDB](https://www.trychroma.com/) 团队提供的易用向量数据库。
*   所有开源社区的贡献者。
