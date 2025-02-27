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
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain のインポート（langchain パッケージがインストールされていることを確認してください）
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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
# 実際の Azure OpenAI 設定に従って以下の変数を記入してください
api_key = "YOUR_AZURE_OPENAI_API_KEY"
api_base = "https://YOUR_RESOURCE_NAME.openai.azure.com/"  # Azure OpenAI リソースのエンドポイント
api_version = "2023-05-15"  # API バージョン（必要に応じて調整してください）
deployment_name = "YOUR_DEPLOYMENT_NAME"  # モデルのデプロイメント名（例: Azure にデプロイした GPT-4 または GPT-3.5 のモデル名）

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

# 4. 各チェック項目に対して、AI評価のコメント生成と結果判定を実行します。LLM を二段階で呼び出します。
def process_check_item(item):
    """各チェック項目に対して、AI評価のコメント生成と結果判定を実行します。LLM を二段階で呼び出します。"""
    item_no = item.get("項番", "<no-id>")
    content = str(item.get("確認内容", ""))  # 確認内容
    try:
        # ステージ1の LLM を呼び出し、AI評価のコメントを生成します
        comment_prompt = prompt_stage1.format(check_item=content, document=doc_text)
        ai_comment = llm([{"role": "user", "content": comment_prompt}]).content  # Chat モデルを呼び出し
        logging.info(f"項番 {item_no}: ステージ1のコメント生成が完了しました")
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ1のコメント生成に失敗しました: {e}")
        # ステージ1が失敗した場合、ステージ2をスキップしてエラー情報を返します
        return {
            "項番": item_no,
            "AI評価のコメント": f"エラー: {e}",
            "AI評価": "-",
            "改善案": ""
        }
    try:
        # ステージ1のコメントに基づき、ステージ2の LLM を呼び出して構造化された評価結果を生成します
        result_prompt = prompt_stage2.format(ai_comment=ai_comment)
        raw_output = llm([{"role": "user", "content": result_prompt}]).content
        # モデルの出力を構造化された結果にパースします
        result = output_parser.parse(raw_output)
        ai_hyouka = result.get("AI評価", "")
        kaizen = result.get("改善案", "")
        logging.info(f"項番 {item_no}: ステージ2の評価生成が完了しました (評価: {ai_hyouka})")
    except Exception as e:
        logging.error(f"項番 {item_no}: ステージ2の評価生成または解析に失敗しました: {e}")
        # ステージ2が失敗した場合、評価を NG とし、改善案にエラー情報を記録します
        ai_hyouka = "-"
        kaizen = f"エラー: {e}"
    # 新しいフィールドを含む辞書を返します
    return {
        "項番": item_no,
        "AI評価のコメント": ai_comment,
        "AI評価": ai_hyouka,
        "改善案": kaizen
    }

# 5. すべてのチェック項目を並列処理します
logging.info("チェック項目の並列処理を開始します")
items = df.to_dict(orient="records")  # DataFrame を辞書のリストに変換し、渡しやすくします
results = []

# スレッドプールを使用して各チェック項目を並列実行します
with ThreadPoolExecutor() as executor:
    # 全タスクを提出します
    futures = [executor.submit(process_check_item, item) for item in items]
    # タスクが完了した順に結果を収集します
    for future in as_completed(futures):
        result = future.result()
        results.append(result)

logging.info("すべてのチェック項目の処理が完了しました")

# 結果のリストを DataFrame に変換し、項番で並べ替えて元の順序と一致させます
result_df = pd.DataFrame(results)
# 項番が数値の場合はソートできます。数値でない場合は元の出現順序に従ってマッチングします。
try:
    result_df["項番"] = result_df["項番"].astype(int)
except:
    pass
if "項番" in df.columns:
    # 結果の DataFrame を項番順に並び替えます
    result_df = result_df.set_index("項番").loc[df["項番"]].reset_index()

# 6. 結果を元の DataFrame に統合し、Excel ファイルに出力します
# AI評価の列を元の DataFrame に統合します
for col in ["AI評価のコメント", "AI評価", "改善案"]:
    df[col] = df["項番"].map(result_df.set_index("項番")[col])

# Excel ファイルとして保存し、日本語の文字が正しく保存されることを確認します
output_file = "checklist.xlsx"
try:
    df.to_excel(output_file, index=False, encoding="utf-8")
    logging.info(f"結果は {output_file} に保存されました")
except Exception as e:
    logging.error(f"Excel の保存中にエラーが発生しました: {e}")
    raise

print("検査が完了しました。結果は", output_file, "に保存されました")


