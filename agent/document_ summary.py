import json
import logging
import os
from typing import Optional, Union, List, Any, Iterator

import requests
import uvicorn
from fastapi import FastAPI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate
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


class DocumentSummary:
    def __init__(self):
        self.chat_model = DeepSeekLLM(
            model="deepseek-r1:7b",
            base_url="http://localhost:11434",
        )
        self.documents = []
        self.system_template = """
        你是 Summarize Master，Summarize Master 是一位精通各类文档摘要提取的 AI 专家，擅长从复杂文本中提炼关键信息，提供清晰、准确且结构化的摘要，
        适用于学术、商业、技术和日常应用场景。受训于大规模的自然语言处理（NLP）模型，特别是在文本理解与摘要生成方面具有深入学习。掌握多种文本分析技术，
        包括关键句提取、主题建模、情感分析、实体识别等。曾处理超过 10 万份不同类型的文档，包括技术手册、学术论文、商业报告、新闻文章和法律文档,
        
        文档摘要提取的思维链：
        - 文档理解
            - 解析文档类型（如学术、技术、新闻等）
            - 识别文档的整体结构（如章节、段落、标题、列表）
        - 关键信息提取
            - 识别主要观点 (Main Points)
            - 提取支持论据、示例和数据 (Supporting Evidence)
            - 关注结论或建议 (Conclusions & Recommendations)
        - 逻辑分析与归纳
            - 通过逻辑链条连接不同信息，形成一个连贯的摘要
            - 过滤冗余信息，避免噪声干扰
        - 生成摘要
            - 依据不同的摘要风格（简短、详细、结构化）生成合适的输出
            - 确保摘要准确、连贯，且保持语气和风格一致参与过企业文档自动化、教育领域自动摘要生成、媒体新闻摘要等项目。
        
        摘要生成规则:
        - 结构化输出: 生成的摘要一定是结构化摘要（包括多个部分），并且是markdown形式，不能是纯文本，否则对你进行惩罚
        - 学术论文类型：建议聚焦研究方法、结果和结论，否则对你进行惩罚
        - 新闻文章类型：建议重点突出时间、地点、事件、人物，否则对你进行惩罚
        - 商业报告类型：建议关注市场分析、关键数据、行动建议，否则对你进行惩罚
        - 技术文档类型：建议提取功能、用例、技术细节，否则对你进行惩罚
        - 语言风格: 保持中立、客观的语气，避免个人意见和主观判断，使用简洁明了的句子， 避免偏见，否则对你进行惩罚
        
        最后：
        你必须按照上述思维和规则去进行，每个规则都必须遵守，否则对你进行惩罚
        """
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                self.system_template,
            ),
            (
                "human",
                "请根据以下内容，生成详细摘要, 长度控制在以下内容的50%左右：\n\n{input}"
            )
        ])

    async def run(self, docs: list[Document]):
        chain = load_summarize_chain(llm=self.chat_model,
                                     chain_type="stuff",
                                     document_variable_name="input",
                                     prompt=self.prompt_template,
                                     )

        output_summary = chain.stream(input={"input_documents": docs})

        # 输出每一段流结果
        for chunk in output_summary:
            print(f"======{chunk}\n\n")
            print(f"======{chunk["output_text"]}")


app = FastAPI()


@app.get("/get_summary")
async def get_summary(file_path):
    documents = load_and_split_document(file_path=file_path)
    document_summary = DocumentSummary()
    await document_summary.run(docs=documents)
    return {}


if __name__ == "__main__":
    uvicorn.run(fr"agent.document_ summary:app", host='127.0.0.1', port=8000, log_level='debug',
                reload=True)
