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
    

# 【修改】新增一个函数，用于批量处理多个 xlsx 文件
def process_uploaded_files(uploaded_excel_files, selected_types, selected_model, llm_instance):

    all_summaries = []
    generated_excels = []

    if not isinstance(uploaded_excel_files, list):
        uploaded_excel_files = [uploaded_excel_files]

    for file_data in uploaded_excel_files:
        summary, excel_path = process_uploaded_file(file_data, selected_types, selected_model, llm_instance)
        all_summaries.append(summary)
        if excel_path:
            generated_excels.append(excel_path)

    combined_summary = "\n\n".join(all_summaries)


    if not generated_excels:
        return combined_summary, None

    zip_name = "batch_results.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for excel_file in generated_excels:
            arcname = os.path.basename(excel_file)
            zf.write(excel_file, arcname)

    return combined_summary, zip_name

def run_interface(uploaded_excel_files, selected_model, selected_types):

    try:
        config = MODEL_CONFIG.get(selected_model)
        if config is None:
            raise ValueError("選択されたモデルが無効です。")

        llm_instance = AzureChatOpenAI(
            deployment_name=config["deployment_name"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0
        )

        # 【修改】批量处理文件
        summary, zip_file = process_uploaded_files(uploaded_excel_files, selected_types, selected_model, llm_instance)
        return summary, zip_file
    except Exception as e:
        return f"全体処理中にエラーが発生しました: {e}", None

# 读取 check_list.csv 中的種別选项，用于 UI 的 CheckboxGroup
try:
    df_check = pd.read_csv("check_list.csv", encoding="utf-8-sig")
    types_options = df_check["種別"].dropna().unique().tolist()
except Exception as e:
    logging.error(f"チェックリストの読み込みに失敗しました: {e}")
    types_options = []

# 【修改】Gradio界面：改为 gr.Files，可以一次上传多个 xlsx
with gr.Blocks() as demo:
    gr.Markdown("## AI 手順書チェックツール (複数ファイル対応版)")
    with gr.Row():
        # 【修改】这里使用 gr.Files 而不是 gr.File，并允许多文件
        input_files = gr.Files(label="Excel ファイルをまとめてアップロード", file_types=[".xlsx", ".xlsm"])
    with gr.Row():
        model_selection = gr.Radio(choices=["GPT3.5", "4omini"], label="モデル選択", value="4omini")
        selected_types = gr.CheckboxGroup(choices=types_options, label="チェック項目の種別選択 (空欄なら全て対象)")
    with gr.Row():
        run_btn = gr.Button("処理開始")
    with gr.Row():
        output_message = gr.Textbox(label="処理情報", interactive=False, lines=15)
    with gr.Row():
        # 这里的 file_types=[".zip"] 便于用户下载一个压缩包
        output_file = gr.File(label="結果の Zip ファイルをダウンロード", file_types=[".zip"])
    
    # 注意：Inputs改为 input_files (list)，Outputs保持两个：summary + zip_file
    run_btn.click(fn=run_interface, inputs=[input_files, model_selection, selected_types], outputs=[output_message, output_file])

demo.launch()

