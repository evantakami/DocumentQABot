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
    

# 【修改】辅助函数，将JSON格式的数据转换成Excel可用的字符串
def safe_to_excel(val):
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return val

            for idx, result in enumerate(file_results):
                excel_row = start_row + idx

                ai_comment_val = safe_to_excel(result.get("AI評価のコメント", ""))
                ai_hyouka_val = safe_to_excel(result.get("AI評価", ""))
                kaizen_val = safe_to_excel(result.get("改善案", ""))

                cell = new_sheet.cell(row=excel_row, column=start_col)
                cell.value = ai_comment_val
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

                cell = new_sheet.cell(row=excel_row, column=start_col + 1)
                cell.value = ai_hyouka_val
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

                cell = new_sheet.cell(row=excel_row, column=start_col + 2)
                cell.value = kaizen_val
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

# MODIFY START: 使用 Streamlit 构建界面
try:
    df_check = pd.read_csv("check_list.csv", encoding="utf-8-sig")
    types_options = df_check["種別"].dropna().unique().tolist()
except Exception as e:
    logging.error(f"チェックリストの読み込みに失敗しました: {e}")
    types_options = []

st.title("AI 手順書チェックツール (複数ファイル対応版)")

st.markdown("### Excel ファイルをまとめてアップロード")
uploaded_files = st.file_uploader("上传Excel文件", type=["xlsx", "xlsm"], accept_multiple_files=True)

st.markdown("### モデル選択")
model_selection = st.radio("モデル選択", options=["GPT3.5", "4omini"], index=1)

st.markdown("### チェック項目の種別選択 (空欄なら全て対象)")
selected_types = st.multiselect("選択してください", options=types_options)

if st.button("処理開始"):
    if not uploaded_files:
        st.error("请先上传至少一个Excel文件")
    else:
        with st.spinner("处理中，请稍候..."):
            summary, zip_file = run_interface(uploaded_files, model_selection, selected_types)
        st.text_area("処理情報", value=summary, height=200)
        if zip_file and os.path.exists(zip_file):
            with open(zip_file, "rb") as f:
                st.download_button("结果的 Zip 文件下载", data=f.read(), file_name=zip_file)
# MODIFY END