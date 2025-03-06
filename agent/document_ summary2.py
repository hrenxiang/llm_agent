import json
import logging
import os
from typing import Optional, Union, List, Any, Iterator

import requests
import uvicorn
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.messages import HumanMessage
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from rapidocr_onnxruntime import RapidOCR

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekLLM(LLM):
    model: str
    """要使用的模型名称"""

    temperature: Optional[float] = None
    """模型的温度。增加温度会使模型的回答更加具有创造性。默认值：0.8"""

    stop: Optional[List[str]] = None
    """设置停止词。生成过程中，如果输出中首次出现这些停止子串，则输出会被截断"""

    tfs_z: Optional[float] = None
    """尾部自由采样，减少输出中低概率词的影响。较高的值（如 2.0）会更有效，而 1.0 禁用此设置。默认值：1"""

    top_k: Optional[int] = None
    """限制生成的词汇数量。较高的值（如 100）会生成更多元的回答，而较低的值（如 10）则会更加保守。默认值：40"""

    top_p: Optional[float] = None
    """与 top_k 配合使用。较高的值（如 0.95）会导致生成更多样的文本，而较低的值（如 0.5）会生成更专注的文本。默认值：0.9"""

    format: str = ""
    """指定输出格式（选项：json）"""

    keep_alive: Optional[Union[int, str]] = None
    """模型在内存中保持加载的时间"""

    base_url: Optional[str] = None
    """模型托管的基本 URL"""

    @property
    def _identifying_params(self) -> dict:
        """返回标识模型的参数字典"""
        return {
            "model_name": "DeepSeekLLM",
        }

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的类型"""
        return "deepseek-r1:32B"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            callbacks: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """在给定的提示文本上调用 LLM（语言模型）

        该方法会将提示传递给模型并返回生成的内容。

        参数:
            prompt: 用于生成内容的提示文本。
            stop: 生成时使用的停止词。当输出中首次出现这些停止子串时，输出将被截断。
            callbacks: 可选的回调管理器，用于处理生成过程中的事件。
            **kwargs: 其他任意关键字参数，通常会传递给模型提供者的 API 调用。

        返回:
            模型生成的文本输出，去除了提示文本。
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "top_k": self.top_k,
            "stream": False,  # 非流式调用，保持与 _stream 一致的 API 结构
            **kwargs,  # 支持传入更多自定义参数
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=data,
                timeout=60  # 设置超时，防止请求挂起
            )
            response.raise_for_status()  # 如果响应状态码不是 2xx，会抛出异常

            result = response.json()
            content = result.get("response", "")

            if callbacks:
                callbacks.on_llm_new_token(content)

            # 检查停止词并截断输出
            if stop:
                for s in stop:
                    if s in content:
                        content = content.split(s)[0]
                        break

            return content

        except requests.RequestException as e:
            if callbacks:
                callbacks.on_llm_error(e)
            raise RuntimeError(f"请求失败: {e}")

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """在给定的提示文本上流式运行 LLM（语言模型）

        该方法支持流式输出，在需要逐步生成内容时使用。

        参数:
            prompt: 用于生成内容的提示文本。
            stop: 生成时使用的停止词。当输出中首次出现这些停止子串时，输出将被截断。
            run_manager: 可选的回调管理器，用于处理生成过程中的事件。
            **kwargs: 其他任意关键字参数，通常会传递给模型提供者的 API 调用。

        返回:
            一个 GenerationChunk（生成块）的迭代器，每次返回一个部分生成结果。
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True  # 强制开启流式输出
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            headers=headers,
            json=data,
            stream=True  # 使用 stream=True 以启用流式请求
        )
        response.raise_for_status()

        # 逐行解析服务器推送事件 (SSE) 或流响应
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("response", "")
                    if run_manager:
                        run_manager.on_llm_new_token(content)

                    # 返回一个 GenerationChunk
                    yield GenerationChunk(text=content)

                    # 检查是否有停止词，并在遇到时停止流
                    if stop and any(s in content for s in stop):
                        break
                except json.JSONDecodeError as e:
                    if run_manager:
                        run_manager.on_llm_error(e)
                    continue


# 负责文档加载和拆分
def load_and_split_document(file_path: str):
    """
    根据文件类型加载文档并拆分成多个 Document 对象
    """
    try:
        ext = os.path.splitext(file_path)[-1].lower()
        logger.info("加载文件：%s, 扩展名：%s", file_path, ext)

        if ext == '.txt':
            return load_txt_splitter(file_path)
        elif ext == '.md':
            return load_md_splitter(file_path)
        elif ext in ['.doc', '.docx']:
            return load_word_splitter(file_path)
        elif ext == '.pdf':
            return load_pdf_splitter(file_path)
        elif ext == '.jpg':
            return load_jpg_splitter(file_path)
        else:
            logger.error("不支持的文件类型：%s", ext)
            return []
    except Exception as e:
        logger.exception("加载并拆分文档时出错：%s", e)
        return []


# 加载 TXT 文件
def load_txt_file(file_path: str) -> list:
    try:
        logger.info("加载TXT文件：%s", file_path)
        loader = UnstructuredLoader(file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载TXT文件时出错：%s", e)
        return []


# 加载 Markdown 文件
def load_md_file(file_path: str) -> list:
    try:
        logger.info("加载Markdown文件：%s", file_path)
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载Markdown文件时出错：%s", e)
        return []


# 加载 Word 文件
def load_word_file(file_path: str) -> list:
    try:
        logger.info("加载Word文件：%s", file_path)
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载Word文件时出错：%s", e)
        return []


# 加载 PDF 文件
def load_pdf_file(file_path: str) -> list:
    try:
        logger.info("加载PDF文件：%s", file_path)
        # 设置 OCR 相关环境变量
        os.environ['OCR_AGENT'] = 'unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract'
        loader = UnstructuredPDFLoader(file_path=file_path, mode="elements", strategy="hi_res")
        docs = loader.load()
        return docs
    except Exception as e:
        logger.exception("加载PDF文件时出错：%s", e)
        return []


# 加载 JPG 文件（通过 OCR 识别）
def load_jpg_file(file_path: str) -> str:
    try:
        logger.info("加载JPG文件：%s", file_path)
        ocr = RapidOCR()
        result, _ = ocr(file_path)
        if result:
            ocr_result = [line[1] for line in result]
            return "\n".join(ocr_result)
        else:
            logger.warning("JPG文件OCR未返回结果：%s", file_path)
            return ""
    except Exception as e:
        logger.exception("加载JPG文件时出错：%s", e)
        return ""


# 预处理文本（例如：转换小写、去除标点等）
def preprocess_text(text: str) -> str:
    try:
        logger.info("预处理文本")
        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text
    except Exception as e:
        logger.exception("预处理文本时出错：%s", e)
        return text


# 分割 TXT 文件
def load_txt_splitter(txt_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分TXT文件：%s", txt_file)
        docs = load_txt_file(txt_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分TXT文件时出错：%s", e)
        return []


# 分割 Markdown 文件
def load_md_splitter(md_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分Markdown文件：%s", md_file)
        docs = load_md_file(md_file)
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分Markdown文件时出错：%s", e)
        return []


# 分割 Word 文件
def load_word_splitter(word_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分Word文件：%s", word_file)
        docs = load_word_file(word_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分Word文件时出错：%s", e)
        return []


# 分割 PDF 文件
def load_pdf_splitter(pdf_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分PDF文件：%s", pdf_file)
        docs = load_pdf_file(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    except Exception as e:
        logger.exception("拆分PDF文件时出错：%s", e)
        return []


# 保存问答的历史记录
store = {}


# 回调函数，此函数预期将接收一个 session_id 并返回一个消息历史记录对象
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 分割 JPG 文件（OCR后处理）
def load_jpg_splitter(jpg_file: str, chunk_size=200, chunk_overlap=20):
    try:
        logger.info("拆分JPG文件：%s", jpg_file)
        docs = load_jpg_file(jpg_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 因为 OCR 返回的是字符串，所以使用 create_documents 将文本转换成 Document 对象
        split_docs = text_splitter.create_documents([docs])
        return split_docs
    except Exception as e:
        logger.exception("拆分JPG文件时出错：%s", e)
        return []


# 在历史记录中存储文件路径和内容
def store_uploaded_file(session_id: str, file_path: str):
    session_history = get_session_history(session_id)
    session_history.add_message(message=HumanMessage(content=f"文件上传: {file_path}"))


@tool
def load_file(file_path: str):
    """实现文件加载逻辑"""
    return load_and_split_document(file_path)


class DocumentSummary:
    def __init__(self):
        self.chat_model = DeepSeekLLM(
            model="deepseek-r1:7b",
            base_url="http://localhost:11434",
        )
        self.MEMORY_KEY = "chat-history"
        self.memory = ""
        self.system_template = """
        你是河南移动开发的AI灵犀助手，此处主要负责文档解读！
        """
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                self.system_template,
            ),
            (
                "human",
                '''TOOLS
                            ------
                            Assistant can ask the user to use tools to look up information that may be helpful in \
                            answering the users original question. The tools the human can use are:
        
                            {tools}
        
                            RESPONSE FORMAT INSTRUCTIONS
                            ----------------------------
        
                            When responding to me, please output a response in one of two formats:
        
                            **Option 1:**
                            Use this if you want the human to use a tool.
                            Markdown code snippet formatted in the following schema:
        
                            ```json
                            {{
                                "action": string, \\ The action to take. Must be one of {tool_names}
                                "action_input": string \\ The input to the action
                            }}
                            ```
        
                            **Option #2:**
                            Use this if you want to respond directly to the human. Markdown code snippet formatted \
                            in the following schema:
        
                            ```json
                            {{
                                "action": "Final Answer",
                                "action_input": string \\ You should put what you all return to use here, with think
                            }}
                            ```
        
                            USER'S INPUT
                            --------------------
                            Here is the user's input (remember to respond with a markdown code snippet of a json \
                            blob with a single action, and NOTHING else):
        
                            {input}'''
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        agent = create_json_chat_agent(
            self.chat_model,
            tools=[load_file],
            prompt=self.prompt,
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=[load_file], verbose=True)

    def run(self, user_input: str, ):
        invoke = self.agent_executor.invoke({"input": user_input, })
        print(invoke)
        return invoke


app = FastAPI()


@app.get("/get_summary")
def get_summary(input: str):
    document_summary = DocumentSummary()
    ret = document_summary.run(user_input=input)


if __name__ == "__main__":
    uvicorn.run(fr"agent.document_ summary2:app", host='127.0.0.1', port=8000, log_level='debug',
                reload=True)
