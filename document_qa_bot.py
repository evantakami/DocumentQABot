import gradio as gr
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from ollama_wrapper import OllamaChatWrapper

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
MODEL_NAME = 'elyza:jp8b'  # 指定されたモデル名

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key="")
ollama_chat = OllamaChatWrapper(base_url="http://localhost:11434", model_name="elyza:jp8b")

#Ragas(rag評価マトリックス)の導入
#Rerank Modelの導入
conversation_history = []
retriever = None
qa_chain = None

def process_uploaded_file(file):
    global retriever, qa_chain
    
    if file is None:
        return "ファイルがアップロードされていません。"
    
    file_path = file.name
    if not os.path.exists(file_path):
        return f"ファイルが見つかりません：{file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    
    documents = [Document(page_content=file_content)]
    
    
    text_splitter = MarkdownTextSplitter()
    docs = text_splitter.split_documents(documents)
    
    print(f"分割後のチャンク数：{len(docs)}")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return f"ドキュメントは {len(docs)} 個のチャンクに分割されました。"

def answer_question(question, use_online):
    global conversation_history, retriever, qa_chain
    
    if retriever is None:
        return "ファイルをアップロードしてください。", "", ""
    
    retrieved_docs = retriever.get_relevant_documents(question)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    history_text = "\n".join([f"Human: {q}\nAI: {a}" for q, a in conversation_history[-3:]])
    
    if use_online:
        prompt = f"Conversation history:\n{history_text}\n\nContext: {context_text}\n\nHuman: {question}\nAI:"
        answer = qa_chain.run(prompt)
    else:
        prompt = f"Conversation history:\n{history_text}\n\nContext: {context_text}\n\nHuman: {question}\nAI:"
        answer = ollama_chat.generate(prompt)
    
    conversation_history.append((question, answer))
    if len(conversation_history) > 10:
        conversation_history.pop(0)

    retrieved_chunks = "\n\n".join([f"Chunk {i+1}:\n{doc.page_content[:200]}..." for i, doc in enumerate(retrieved_docs)])
    
    return answer, retrieved_chunks, format_history()

def format_history():
    return "\n\n".join([f"Human: {q}\nAI: {a}" for q, a in conversation_history])

def create_interface():
    with gr.Blocks(css="""
    body {
        background-color: #f0f2f5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .gr-button {
        padding: 10px 20px;
        font-size: 14px;
        background-color: #f57c00;
        color: white;
        border-radius: 5px;
    }
    """) as iface:
        gr.Markdown("# 📄 **DocumentQABot**")
        gr.Markdown("📁 **ファイルをアップロードして会話しましょう！💬**")
        
        with gr.Row():
            file_upload = gr.File(label="📄 **Markdownファイルをアップロード**")
        
        with gr.Row():
            process_btn = gr.Button("📥 **ファイルを処理**")
            process_output = gr.Textbox(label="✅ **処理結果**", scale=2)

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(lines=2, placeholder="❓ **問題を入力してください**")
                use_online = gr.Checkbox(label="🌐 **オンラインモデルを使用する (gpt-4o-mini)**", value=True)
                submit_btn = gr.Button("🚀 **問題を送信**")
                answer_output = gr.Textbox(lines=10, label="📝 **回答**")
            
            with gr.Column(scale=1):
                history_output = gr.Textbox(lines=20, label="📚 **対話履歴**", value="")
        
        retrieved_chunks_output = gr.Textbox(lines=5, label="🔍 **検索されたChunks**")
        
        process_btn.click(fn=process_uploaded_file, inputs=[file_upload], outputs=[process_output])
        submit_btn.click(fn=answer_question, inputs=[question_input, use_online], outputs=[answer_output, retrieved_chunks_output, history_output])


    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(share=True)
import os
import logging
import pandas as pd
import tiktoken  # 用于 token 计数
import time  # 用于等待
import threading  # 用于第一次日志记录的锁
import openpyxl  # 用于操作 Excel

# LangChain のインポート（langchain パッケージがインストールされていることを確認してください）
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import HumanMessage

# --- Token 数を計算する関数 ---
def count_tokens(text, model="gpt-3.5-turbo"):
    """
    指定したテキストの token 数を返します。
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# 全局变量及锁，用于只记录第一次的 Prompt 和 AI 的 comment
first_log_done = False
first_log_lock = threading.Lock()

# 1. データの読み込み: チェックリスト CSV と事前検証 MD ファイルを読み込みます
csv_file = "checklist.csv"
md_file = "事前検証.md"

# ログの設定: INFO レベル以上のログをファイルに記録し、日本語が文字化けしないようにします
logging.basicConfig(filename="process.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")
logging.info("データの読み込みを開始します")

# チェックリスト CSV を読み込みます（UTF-8 エンコードで読み込みを試みます）
try:
    df = pd.read_csv(csv_file, encoding="utf-8-sig")  # utf-8-sig を使用して BOM をサポート
except Exception as e:
    logging.error(f"CSV ファイルの読み込みに失敗しました: {e}")
    raise

# Markdown ドキュメントの内容を読み込みます
try:
    with open(md_file, "r", encoding="utf-8") as f:
        doc_text = f.read()
except Exception as e:
    logging.error(f"Markdown ファイルの読み込みに失敗しました: {e}")
    raise

logging.info(f"チェック項目の数: {len(df)}")
logging.info("データの読み込みが完了しました")

# 2. Azure OpenAI の設定: API 認証情報とモデルデプロイメント名を設定します
api_key = "YOUR_AZURE_OPENAI_API_KEY"
api_base = "https://YOUR_RESOURCE_NAME.openai.azure.com/"  # Azure OpenAI リソースのエンドポイント
api_version = "2023-05-15"  # API バージョン
deployment_name = "YOUR_DEPLOYMENT_NAME"  # モデルのデプロイメント名

# 用户选择的模型（费用计算依据），例如 "gpt-3.5-turbo" 或 "gpt-4"
selected_model = "gpt-3.5-turbo"

# 设置费用（单位：美元/Token），这里仅为示例，具体定价请参照实际情况
cost_dict = {
    "gpt-3.5-turbo": 0.002 / 1000,
    "gpt-4": 0.03 / 1000
}

# 認証情報を環境変数に設定します（LangChain はデフォルトで環境変数から読み込みます）
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_VERSION"] = api_version

# AzureChatOpenAI オブジェクトを初期化します
try:
    llm = AzureChatOpenAI(deployment_name=deployment_name,
                          openai_api_base=api_base,
                          openai_api_version=api_version,
                          openai_api_key=api_key,
                          openai_api_type="azure",
                          temperature=0)
    logging.info("Azure OpenAI LLM の初期化に成功しました")
except Exception as e:
    logging.error(f"Azure OpenAI の初期化に失敗しました: {e}")
    raise

# 3. 多段階検査に必要なプロンプトテンプレートとパーサーを定義します

# ステージ1プロンプトテンプレート：チェック項目と文書全体に基づいて、AI評価のコメントを生成します
template_stage1 = """あなたは手順書の事前検証をチェックする高度なAIアシスタントです。
以下のチェック項目と文書内容に基づいて、その項目に関するAI評価のコメントを日本語で述べてください。
チェック項目: {check_item}
文書内容: \"\"\"{document}\"\"\"
AI評価のコメント:"""
prompt_stage1 = PromptTemplate(
    input_variables=["check_item", "document"],
    template=template_stage1
)

# ステージ2の出力パーサー: JSON 形式で「AI評価」と「改善案」を出力する構造を定義します
response_schemas = [
    ResponseSchema(name="AI評価", description="評価結果。OK、NG、または「-」のいずれか。"),
    ResponseSchema(name="改善案", description="評価がNGの場合、項目を満たすための改善提案。OKや「-」の場合は空文字。")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# ステージ2プロンプトテンプレート：ステージ1のコメントに基づき、構造化された評価結果と改善案を出力します
template_stage2 = """あなたはチェック結果を判定するアシスタントです。
以下のAI評価コメントに基づき、チェック項目の最終評価と改善案を決定してください。
評価は「OK」「NG」または「-」で表し、NGの場合は改善案も提案してください。
{format_instructions}
AI評価コメント: {ai_comment}"""
prompt_stage2 = PromptTemplate(
    input_variables=["ai_comment"],
    partial_variables={"format_instructions": format_instructions},
    template=template_stage2
)

def process_check_item(item):
    """
    各チェック項目に対して、二段階でLLMを呼び出し、
    「AI評価のコメント」と「AI評価」および「改善案」を生成し、さらに Token 消費数も計算して返します。
    这里增加 total_tokens 字段用于费用统计，但后续输出Excel时不包含 Token 消耗列。
    """
    global first_log_done
    item_no = item.get("項番", "<no-id>")
    content = str(item.get("確認内容", ""))
    
    # --- ステージ1: AI評価のコメント生成 ---
    try:
        comment_prompt = prompt_stage1.format(check_item=content, document=doc_text)
        ai_comment = llm([HumanMessage(content=comment_prompt)]).content
        logging.info(f"項番 {item_no}: ステージ1のコメント生成が完了しました")
        tokens_prompt1 = count_tokens(comment_prompt)
        tokens_response1 = count_tokens(ai_comment)
        
        # 仅在第一次处理时，将用户给的 prompt 和 AI 的 comment 追加到日志中
        with first_log_lock:
            if not first_log_done:
                logging.info("User给的Prompt内容: " + comment_prompt)
                logging.info("AI的comment内容: " + ai_comment)
                first_log_done = True
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ1のコメント生成に失敗しました: {e}")
        return {
            "項番": item_no,
            "AI評価のコメント": f"エラー: {e}",
            "AI評価": "-",
            "改善案": "",
            "Token消費": f"エラー: {e}",
            "total_tokens": 0
        }
    
    # --- ステージ2: 評価結果と改善案の生成 ---
    try:
        result_prompt = prompt_stage2.format(ai_comment=ai_comment)
        raw_output = llm([HumanMessage(content=result_prompt)]).content
        result = output_parser.parse(raw_output)
        ai_hyouka = result.get("AI評価", "")
        kaizen = result.get("改善案", "")
        logging.info(f"項番 {item_no}: ステージ2の評価生成が完了しました (評価: {ai_hyouka})")
        tokens_prompt2 = count_tokens(result_prompt)
        tokens_response2 = count_tokens(raw_output)
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ2の評価生成または解析に失敗しました: {e}")
        ai_hyouka = "-"
        kaizen = f"エラー: {e}"
        tokens_prompt2 = 0
        tokens_response2 = 0
    
    total_tokens = tokens_prompt1 + tokens_response1 + tokens_prompt2 + tokens_response2
    token_info = (f"Stage1: prompt {tokens_prompt1}, response {tokens_response1}; "
                  f"Stage2: prompt {tokens_prompt2}, response {tokens_response2}; "
                  f"合計: {total_tokens}")
    
    return {
        "項番": item_no,
        "AI評価のコメント": ai_comment,
        "AI評価": ai_hyouka,
        "改善案": kaizen,
        "Token消費": token_info,
        "total_tokens": total_tokens
    }

# 5. すべてのチェック項目を順次処理します（取消并行处理）
logging.info("チェック項目の処理を順次実行します")
items = df.to_dict(orient="records")
results = []
total_tasks = len(items)
total_tokens_sum = 0

for idx, item in enumerate(items):
    result = process_check_item(item)
    results.append(result)
    total_tokens_sum += result.get("total_tokens", 0)
    print(f"進捗: {idx+1}/{total_tasks} 完了")
    # 每个任务间可适当等待（如需）：
    time.sleep(1)

# 根据用户选择的模型进行费用计算（不反映在Excel上，仅记录日志）
total_cost = total_tokens_sum * cost_dict.get(selected_model, 0)
logging.info(f"使用模型 {selected_model} 総Token消費: {total_tokens_sum}, 费用: ${total_cost:.6f}")

# 6. 結果を元の DataFrame に統合し、Excel ファイル（原Excelのコピー）に出力します
# 这里只取需要的3个字段，不包含 Token消費 列
for col in ["AI評価のコメント", "AI評価", "改善案"]:
    df[col] = df["項番"].map({r["項番"]: r[col] for r in results})

# 原Excel文件名称
original_excel = "F-0168-2.xlsx"
# 处理后Excel文件名称（保留原文件，仅对复制文件进行处理）
output_excel = "F-0168-2_Processed.xlsx"

# 读取原Excel文件，并定位到指定工作表
wb = openpyxl.load_workbook(original_excel)
ws = wb["商用作業手順書チェックリスト"]

# 在 H列(列8)和 J列(列10)之间插入3列（即在第9列开始插入3列）
ws.insert_cols(idx=9, amount=3)

# 定义新插入列的标题（标题行在第8行），并复制 H列（列8）单元格的格式
new_headers = ["AI評価のコメント", "AI評価", "改善案"]

def copy_cell_style(source, target):
    target.font = source.font
    target.border = source.border
    target.fill = source.fill
    target.number_format = source.number_format
    target.protection = source.protection
    target.alignment = source.alignment

# 复制标题行格式（假设 H8 有合适格式）
source_cell = ws.cell(row=8, column=8)
for offset, header in enumerate(new_headers, start=0):
    cell = ws.cell(row=8, column=9+offset)
    cell.value = header
    copy_cell_style(source_cell, cell)

# 假设Excel中的数据从第9行开始，与CSV中数据行顺序一致
# 将处理结果写入新插入的3列（第9列~第11列），并复制对应行H列的格式
start_row = 9
for i, row_data in enumerate(results):
    excel_row = start_row + i
    # 处理结果字段依次写入列 9, 10, 11
    for col_offset, field in enumerate(new_headers, start=0):
        cell = ws.cell(row=excel_row, column=9+col_offset)
        cell.value = row_data[field]
        source_cell = ws.cell(row=excel_row, column=8)  # 复制H列格式
        copy_cell_style(source_cell, cell)

# 保存处理后的Excel文件
wb.save(output_excel)
logging.info(f"結果は {output_excel} に保存されました")

print("検査が完了しました。結果は", output_excel, "に保存されました")
import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken  # 用于 token 计数
import time  # 用于等待

# LangChain のインポート（langchain パッケージがインストールされていることを確認してください）
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import HumanMessage

# --- Token 数を計算する関数 ---
def count_tokens(text, model="gpt-3.5-turbo"):
    """
    指定したテキストの token 数を返します。
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# 1. データの読み込み: チェックリスト CSV と事前検証 MD ファイルを読み込みます
csv_file = "checklist.csv"
md_file = "事前検証.md"

# ログの設定: INFO レベル以上のログをファイルに記録し、日本語が文字化けしないようにします
logging.basicConfig(filename="process.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")
logging.info("データの読み込みを開始します")

# チェックリスト CSV を読み込みます（UTF-8 エンコードで読み込みを試みます）
try:
    df = pd.read_csv(csv_file, encoding="utf-8-sig")  # utf-8-sig を使用して BOM をサポート
except Exception as e:
    logging.error(f"CSV ファイルの読み込みに失敗しました: {e}")
    raise

# Markdown ドキュメントの内容を読み込みます
try:
    with open(md_file, "r", encoding="utf-8") as f:
        doc_text = f.read()
except Exception as e:
    logging.error(f"Markdown ファイルの読み込みに失敗しました: {e}")
    raise

logging.info(f"チェック項目の数: {len(df)}")
logging.info("データの読み込みが完了しました")

# 2. Azure OpenAI の設定: API 認証情報とモデルデプロイメント名を設定します
api_key = "YOUR_AZURE_OPENAI_API_KEY"
api_base = "https://YOUR_RESOURCE_NAME.openai.azure.com/"  # Azure OpenAI リソースのエンドポイント
api_version = "2023-05-15"  # API バージョン
deployment_name = "YOUR_DEPLOYMENT_NAME"  # モデルのデプロイメント名

# 認証情報を環境変数に設定します（LangChain はデフォルトで環境変数から読み込みます）
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_VERSION"] = api_version

# AzureChatOpenAI オブジェクトを初期化します
try:
    llm = AzureChatOpenAI(deployment_name=deployment_name,
                          openai_api_base=api_base,
                          openai_api_version=api_version,
                          openai_api_key=api_key,
                          openai_api_type="azure",
                          temperature=0)
    logging.info("Azure OpenAI LLM の初期化に成功しました")
except Exception as e:
    logging.error(f"Azure OpenAI の初期化に失敗しました: {e}")
    raise

# 3. 多段階検査に必要なプロンプトテンプレートとパーサーを定義します

# ステージ1プロンプトテンプレート：チェック項目と文書全体に基づいて、AI評価のコメントを生成します
template_stage1 = """あなたは手順書の事前検証をチェックする高度なAIアシスタントです。
以下のチェック項目と文書内容に基づいて、その項目に関するAI評価のコメントを日本語で述べてください。
チェック項目: {check_item}
文書内容: \"\"\"{document}\"\"\"
AI評価のコメント:"""
prompt_stage1 = PromptTemplate(
    input_variables=["check_item", "document"],
    template=template_stage1
)

# ステージ2の出力パーサー: JSON 形式で「AI評価」と「改善案」を出力する構造を定義します
response_schemas = [
    ResponseSchema(name="AI評価", description="評価結果。OK、NG、または「-」のいずれか。"),
    ResponseSchema(name="改善案", description="評価がNGの場合、項目を満たすための改善提案。OKや「-」の場合は空文字。")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# ステージ2プロンプトテンプレート：ステージ1のコメントに基づき、構造化された評価結果と改善案を出力します
template_stage2 = """あなたはチェック結果を判定するアシスタントです。
以下のAI評価コメントに基づき、チェック項目の最終評価と改善案を決定してください。
評価は「OK」「NG」または「-」で表し、NGの場合は改善案も提案してください。
{format_instructions}
AI評価コメント: {ai_comment}"""
prompt_stage2 = PromptTemplate(
    input_variables=["ai_comment"],
    partial_variables={"format_instructions": format_instructions},
    template=template_stage2
)

def process_check_item(item):
    """
    各チェック項目に対して、二段階でLLMを呼び出し、
    「AI評価のコメント」と「AI評価」および「改善案」を生成し、さらに Token 消費数も計算して返します。
    """
    item_no = item.get("項番", "<no-id>")
    content = str(item.get("確認内容", ""))
    
    # --- ステージ1: AI評価のコメント生成 ---
    try:
        comment_prompt = prompt_stage1.format(check_item=content, document=doc_text)
        ai_comment = llm([HumanMessage(content=comment_prompt)]).content
        logging.info(f"項番 {item_no}: ステージ1のコメント生成が完了しました")
        tokens_prompt1 = count_tokens(comment_prompt)
        tokens_response1 = count_tokens(ai_comment)
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ1のコメント生成に失敗しました: {e}")
        return {
            "項番": item_no,
            "AI評価のコメント": f"エラー: {e}",
            "AI評価": "-",
            "改善案": "",
            "Token消費": f"エラー: {e}"
        }
    
    # --- ステージ2: 評価結果と改善案の生成 ---
    try:
        result_prompt = prompt_stage2.format(ai_comment=ai_comment)
        raw_output = llm([HumanMessage(content=result_prompt)]).content
        result = output_parser.parse(raw_output)
        ai_hyouka = result.get("AI評価", "")
        kaizen = result.get("改善案", "")
        logging.info(f"項番 {item_no}: ステージ2の評価生成が完了しました (評価: {ai_hyouka})")
        tokens_prompt2 = count_tokens(result_prompt)
        tokens_response2 = count_tokens(raw_output)
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ2の評価生成または解析に失敗しました: {e}")
        ai_hyouka = "-"
        kaizen = f"エラー: {e}"
        tokens_prompt2 = 0
        tokens_response2 = 0
    
    total_tokens = tokens_prompt1 + tokens_response1 + tokens_prompt2 + tokens_response2
    token_info = (f"Stage1: prompt {tokens_prompt1}, response {tokens_response1}; "
                  f"Stage2: prompt {tokens_prompt2}, response {tokens_response2}; "
                  f"合計: {total_tokens}")
    
    return {
        "項番": item_no,
        "AI評価のコメント": ai_comment,
        "AI評価": ai_hyouka,
        "改善案": kaizen,
        "Token消費": token_info
    }

# 5. すべてのチェック項目を分割バッチで処理します
logging.info("チェック項目の並列処理を開始します")
items = df.to_dict(orient="records")
results = []
total_tasks = len(items)
processed_count = 0
batch_index = 0

while processed_count < total_tasks:
    # バッチサイズの決定：初回は3個、以降は5個ずつ
    if batch_index == 0:
        batch_size = min(3, total_tasks - processed_count)
    else:
        batch_size = min(5, total_tasks - processed_count)
    
    batch_items = items[processed_count: processed_count + batch_size]
    logging.info(f"バッチ {batch_index+1} を開始します（タスク数: {batch_size}）")
    
    batch_completed = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_check_item, item) for item in batch_items]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            batch_completed += 1
            current_progress = processed_count + batch_completed
            print(f"進捗: {current_progress}/{total_tasks} 完了")
    
    processed_count += batch_size
    if processed_count < total_tasks:
        print("次のバッチ処理まで20秒待機中...")
        time.sleep(20)
    
    batch_index += 1

logging.info("すべてのチェック項目の処理が完了しました")

result_df = pd.DataFrame(results)
try:
    result_df["項番"] = result_df["項番"].astype(int)
except:
    pass
if "項番" in df.columns:
    result_df = result_df.set_index("項番").loc[df["項番"]].reset_index()

# 6. 結果を元の DataFrame に統合し、Excel ファイルに出力します
for col in ["AI評価のコメント", "AI評価", "改善案", "Token消費"]:
    df[col] = df["項番"].map(result_df.set_index("項番")[col])

output_file = "checklist.xlsx"
try:
    df.to_excel(output_file, index=False, encoding="utf-8")
    logging.info(f"結果は {output_file} に保存されました")
except Exception as e:
    logging.error(f"Excel の保存中にエラーが発生しました: {e}")
    raise

print("検査が完了しました。結果は", output_file, "に保存されました")

# 读取原 Excel 文件，并定位到指定工作表
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# 假定 original_excel 为原 Excel 文件路径
wb = openpyxl.load_workbook(original_excel)
ws = wb["商用作業手順書チェックリスト"]

# 在 H 列（列8）和 J 列（列10）之间插入 3 列（即在第9列开始插入 3 列）
ws.insert_cols(idx=9, amount=3)

# 定义新插入列的标题（标题在原合并的第8、9行中）
new_headers = ["AI評価のコメント", "AI評価", "改善案"]

# 定义标题单元格样式（仅限于新插入的3列）
header_font = Font(name="Meiryo UI", size=11, bold=True)
header_alignment = Alignment(horizontal="center", vertical="center")
header_fill = PatternFill(fill_type="solid", fgColor="CCCCCC")
header_border = Border(bottom=Side(style="double", color="000000"))

# 定义数据单元格样式
data_font = Font(name="Meiryo UI", size=11, bold=False)
data_alignment = Alignment(horizontal="left", vertical="bottom")  # 不居中
data_fill = PatternFill(fill_type="solid", fgColor="FFFFFF")

# 在新插入区域为每列设置标题（合并第8和第9行单元格），并应用标题样式
for offset, header in enumerate(new_headers):
    col_idx = 9 + offset
    ws.merge_cells(start_row=8, start_column=col_idx, end_row=9, end_column=col_idx)
    cell = ws.cell(row=8, column=col_idx)
    cell.value = header
    cell.font = header_font
    cell.alignment = header_alignment
    cell.fill = header_fill
    cell.border = header_border

# 设置新插入区域的列宽（单位为字符宽度），用户可在代码中手动调整
new_column_widths = {9: 40, 10: 20, 11: 50}  # 示例宽度
for col_idx, width in new_column_widths.items():
    col_letter = get_column_letter(col_idx)
    ws.column_dimensions[col_letter].width = width

# 假设 Excel 中的数据从第10行开始，与 CSV 中数据行顺序一致
start_row = 10
for i, row_data in enumerate(results):
    excel_row = start_row + i
    for col_offset, field in enumerate(new_headers):
        cell = ws.cell(row=excel_row, column=9 + col_offset)
        cell.value = row_data[field]
        cell.font = data_font
        cell.alignment = data_alignment
        cell.fill = data_fill

# 保存修改后的 Excel 文件，确保 output_excel 为目标文件路径
wb.save(output_excel)

import os
import logging
import pandas as pd
import tiktoken  # 用于 token 计数
import time  # 用于等待
import threading  # 用于第一次日志记录的锁
import openpyxl  # 用于操作 Excel
from openpyxl.styles import Font  # 用于手动设置字体
from openpyxl.utils import get_column_letter  # 用于获取列字母

# LangChain のインポート（请确保已安装 langchain 包）
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import HumanMessage

# --- Token 数を計算する関数 ---
def count_tokens(text, model="gpt-3.5-turbo"):
    """
    指定したテキストの token 数を返します。
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# 全局变量及锁，用于只记录第一次的 Prompt 和 AI 的 comment
first_log_done = False
first_log_lock = threading.Lock()

# 1. データの読み込み: チェックリスト CSV と事前検証 MD ファイルを読み込みます
csv_file = "checklist.csv"
md_file = "事前検証.md"

# 日志设置：INFO 级以上的日志记录到文件，保证日语不乱码
logging.basicConfig(filename="process.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")
logging.info("データの読み込みを開始します")

# 读取 CSV（使用 utf-8-sig 支持 BOM）
try:
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
except Exception as e:
    logging.error(f"CSV ファイルの読み込みに失敗しました: {e}")
    raise

# 读取 Markdown 文档内容
try:
    with open(md_file, "r", encoding="utf-8") as f:
        doc_text = f.read()
except Exception as e:
    logging.error(f"Markdown ファイルの読み込みに失敗しました: {e}")
    raise

logging.info(f"チェック項目の数: {len(df)}")
logging.info("データの読み込みが完了しました")

# 2. Azure OpenAI 的设置
api_key = "YOUR_AZURE_OPENAI_API_KEY"
api_base = "https://YOUR_RESOURCE_NAME.openai.azure.com/"  # Azure OpenAI 资源的端点
api_version = "2023-05-15"  # API 版本
deployment_name = "YOUR_DEPLOYMENT_NAME"  # 模型部署名称

# 用户选择的模型（费用计算依据），例如 "gpt-3.5-turbo" 或 "gpt-4"
selected_model = "gpt-3.5-turbo"

# 设置费用（单位：美元/Token），示例数值，请根据实际定价调整
cost_dict = {
    "gpt-3.5-turbo": 0.002 / 1000,
    "gpt-4": 0.03 / 1000
}

# 设置环境变量供 LangChain 使用
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_VERSION"] = api_version

# 初始化 AzureChatOpenAI 对象
try:
    llm = AzureChatOpenAI(deployment_name=deployment_name,
                          openai_api_base=api_base,
                          openai_api_version=api_version,
                          openai_api_key=api_key,
                          openai_api_type="azure",
                          temperature=0)
    logging.info("Azure OpenAI LLM の初期化に成功しました")
except Exception as e:
    logging.error(f"Azure OpenAI の初期化に失敗しました: {e}")
    raise

# 3. 定义多阶段检查所需的 Prompt 模板和输出解析器
# --- 修改部分：Stage1 Prompt 增加“手順作成者コメント” ---
template_stage1 = """あなたは手順書の事前検証をチェックする高度なAIアシスタントです。
以下の情報に基づいて、その項目に関するAI評価のコメントを日本語で述べてください。

【チェック項目】
- 種別: {type_main}
- 種別_小: {type_sub}
- 確認内容: {check_item}
- 実施例及び注意観点など: {example}
- 手順作成者コメント: {creator_comment}

【文書内容】
\"\"\"{document}\"\"\"

AI評価のコメント:"""

prompt_stage1 = PromptTemplate(
    input_variables=["type_main", "type_sub", "check_item", "example", "creator_comment", "document"],
    template=template_stage1
)

# 阶段2输出结构
response_schemas = [
    ResponseSchema(name="AI評価", description="評価結果。OK、NG、または「-」のいずれか。"),
    ResponseSchema(name="改善案", description="評価がNGの場合、項目を満たすための改善提案。OKや「-」の場合は空文字。")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template_stage2 = """あなたはチェック結果を判定するアシスタントです。
以下のAI評価コメントに基づき、チェック項目の最終評価と改善案を決定してください。
評価は「OK」「NG」または「-」で表し、NGの場合は改善案も提案してください。
{format_instructions}
AI評価コメント: {ai_comment}"""
prompt_stage2 = PromptTemplate(
    input_variables=["ai_comment"],
    partial_variables={"format_instructions": format_instructions},
    template=template_stage2
)

def process_check_item(item):
    """
    对每个检查项，调用 LLM 进行两阶段处理，
    生成「AI評価のコメント」、「AI評価」和「改善案」，并计算 Token 消耗（用于费用计算，不输出到 Excel）。
    """
    global first_log_done

    item_no = item.get("項番", "<no-id>")
    # 从 CSV 中提取各字段，若为空则为“”
    type_main = str(item.get("種別", ""))
    type_sub = str(item.get("種別_小", ""))
    check_item_text = str(item.get("確認内容", ""))
    example = str(item.get("実施例及び注意観点など", ""))
    creator_comment = str(item.get("手順作成者コメント", ""))

    # --- 阶段1：生成 AI评价评论 ---
    try:
        comment_prompt = prompt_stage1.format(
            type_main=type_main,
            type_sub=type_sub,
            check_item=check_item_text,
            example=example,
            creator_comment=creator_comment,
            document=doc_text
        )
        ai_comment = llm([HumanMessage(content=comment_prompt)]).content
        logging.info(f"項番 {item_no}: ステージ1のコメント生成が完了しました")
        tokens_prompt1 = count_tokens(comment_prompt, model=selected_model)
        tokens_response1 = count_tokens(ai_comment, model=selected_model)
        
        # 仅第一次处理时，将用户给的 prompt 和 AI 的 comment 写入日志
        with first_log_lock:
            if not first_log_done:
                logging.info("User给的Prompt内容: " + comment_prompt)
                logging.info("AI的comment内容: " + ai_comment)
                first_log_done = True
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ1のコメント生成に失敗しました: {e}")
        return {
            "項番": item_no,
            "AI評価のコメント": f"エラー: {e}",
            "AI評価": "-",
            "改善案": "",
            "Token消費": f"エラー: {e}",
            "total_tokens": 0
        }
    
    # --- 阶段2：生成评价结果与改善案 ---
    try:
        result_prompt = template_stage2.format(ai_comment=ai_comment)
        raw_output = llm([HumanMessage(content=result_prompt)]).content
        result = output_parser.parse(raw_output)
        ai_hyouka = result.get("AI評価", "")
        kaizen = result.get("改善案", "")
        logging.info(f"項番 {item_no}: ステージ2の評価生成が完了しました (評価: {ai_hyouka})")
        tokens_prompt2 = count_tokens(result_prompt, model=selected_model)
        tokens_response2 = count_tokens(raw_output, model=selected_model)
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ2の評価生成または解析に失敗しました: {e}")
        ai_hyouka = "-"
        kaizen = f"エラー: {e}"
        tokens_prompt2 = 0
        tokens_response2 = 0
    
    total_tokens = tokens_prompt1 + tokens_response1 + tokens_prompt2 + tokens_response2
    token_info = (f"Stage1: prompt {tokens_prompt1}, response {tokens_response1}; "
                  f"Stage2: prompt {tokens_prompt2}, response {tokens_response2}; "
                  f"合計: {total_tokens}")
    
    return {
        "項番": item_no,
        "AI評価のコメント": ai_comment,
        "AI評価": ai_hyouka,
        "改善案": kaizen,
        "Token消費": token_info,
        "total_tokens": total_tokens
    }

# 5. 顺序处理所有检查项（取消并行处理），每处理一项后等待 1 秒
logging.info("チェック項目の処理を順次実行します")
items = df.to_dict(orient="records")
results = []
total_tasks = len(items)
total_tokens_sum = 0

for idx, item in enumerate(items):
    result = process_check_item(item)
    results.append(result)
    total_tokens_sum += result.get("total_tokens", 0)
    print(f"進捗: {idx+1}/{total_tasks} 完了")
    time.sleep(1)

# 根据用户选择的模型计算费用（仅写入日志，不输出到 Excel）
total_cost = total_tokens_sum * cost_dict.get(selected_model, 0)
logging.info(f"使用模型 {selected_model} 総Token消費: {total_tokens_sum}, 费用: ${total_cost:.6f}")

# 6. 将结果合并到 DataFrame 中，仅保留需要的三个字段，不包含 Token 消耗列
for col in ["AI評価のコメント", "AI評価", "改善案"]:
    df[col] = df["項番"].map({r["項番"]: r[col] for r in results})

# 原 Excel 文件名称（保留原文件，仅对复制文件进行处理）
original_excel = "F-0168-2.xlsx"
output_excel = "F-0168-2_Processed.xlsx"

# 读取原 Excel 文件，并定位到指定工作表
import openpyxl
from openpyxl.utils import get_column_letter
wb = openpyxl.load_workbook(original_excel)
ws = wb["商用作業手順書チェックリスト"]

# 在 H 列（列8）和 J 列（列10）之间插入 3 列（即在第9列开始插入 3 列）
ws.insert_cols(idx=9, amount=3)

# 定义新插入列的标题（标题在原合并的第8、9行中）
new_headers = ["AI評価のコメント", "AI評価", "改善案"]

# 手动设定标题和数据的字体
from openpyxl.styles import Font
header_font = Font(name="Microsoft YaHei", size=11, bold=True)
data_font = Font(name="Microsoft YaHei", size=11, bold=False)

# 在新插入区域为每列设置标题（合并第8和第9行单元格），直接设定字体
for offset, header in enumerate(new_headers, start=0):
    col_idx = 9 + offset
    ws.merge_cells(start_row=8, start_column=col_idx, end_row=9, end_column=col_idx)
    cell = ws.cell(row=8, column=col_idx)
    cell.value = header
    cell.font = header_font

# 设置新插入区域的列宽（宽度可根据需要调整，单位为字符宽度）
new_column_widths = {9: 40, 10: 20, 11: 50}  # 示例宽度
for col_idx, width in new_column_widths.items():
    col_letter = get_column_letter(col_idx)
    ws.column_dimensions[col_letter].width = width

# 假设 Excel 中的数据从第10行开始，与 CSV 中数据行顺序一致
start_row = 10
for i, row_data in enumerate(results):
    excel_row = start_row + i
    for col_offset, field in enumerate(new_headers, start=0):
        cell = ws.cell(row=excel_row, column=9+col_offset)
        cell.value = row_data[field]
        cell.font = data_font

# 保存处理后的 Excel 文件
wb.save(output_excel)
logging.info(f"結果は {output_excel} に保存されました")

print("検査が完了しました。結果は", output_excel, "に保存されました")

