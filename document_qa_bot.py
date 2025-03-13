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
        # 【恢复】在同一个 Workbook 中，为每个 md 文件创建一张 sheet
        results_dict = {}
        cost_summary = {}
        for md_filename in md_files:
            md_path = os.path.join(md_folder, md_filename)
            with open(md_path, "r", encoding="utf-8") as f:
                doc_text = f.read()

            file_results = []
            total_input_tokens = 0
            total_output_tokens = 0

            # 逐行处理 CSV
            for item in items:
                # 如果「種別」不在 selected_types 且 selected_types 非空，则写“チェック対象外”
                if selected_types and item.get("種別", "") not in selected_types:
                    result = {
                        "項番": item.get("項番", ""),
                        "AI評価のコメント": "チェック対象外",
                        "AI評価": "",
                        "改善案": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                else:
                    # 调用 GPT 进行检查
                    result = process_check_item(item, doc_text, llm_instance, selected_model)

                file_results.append(result)
                total_input_tokens += result.get("input_tokens", 0)
                total_output_tokens += result.get("output_tokens", 0)
                time.sleep(2)  # API调用间隔

            results_dict[md_filename] = file_results

            # 统计 token 费用
            model_cost = MODEL_CONFIG.get(selected_model, {"input_cost": 0, "output_cost": 0})
            cost_input = total_input_tokens * model_cost["input_cost"]
            cost_output = total_output_tokens * model_cost["output_cost"]
            total_cost = cost_input + cost_output
            cost_summary[md_filename] = (total_input_tokens, total_output_tokens, total_cost)

            # 【恢复】复制模板工作表并重命名为 md 文件名（去掉后缀）
            new_sheet = wb.copy_worksheet(template_sheet)
            sheet_title = os.path.splitext(md_filename)[0]
            # Excel 的 sheet title 有长度和字符限制，必要时可截断
            new_sheet.title = sheet_title[:31]

            # 写入数据
            data_font = Font(name="Meiryo UI", size=10, bold=False)
            data_alignment = Alignment(horizontal="left", vertical="bottom")
            data_fill = PatternFill(fill_type="solid", fgColor="FFFFFF")
            start_row = 10  # CSV第一行对应 Excel 第10行
            start_col = 11  # 从第11列开始写入

            for idx, result in enumerate(file_results):
                excel_row = start_row + idx
                # AI評価のコメント
                cell = new_sheet.cell(row=excel_row, column=start_col)
                cell.value = result.get("AI評価のコメント", "")
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

                # AI評価
                cell = new_sheet.cell(row=excel_row, column=start_col + 1)
                cell.value = result.get("AI評価", "")
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

                # 改善案
                cell = new_sheet.cell(row=excel_row, column=start_col + 2)
                cell.value = result.get("改善案", "")
                cell.font = data_font
                cell.alignment = data_alignment
                cell.fill = data_fill

        # 【恢复】循环结束后，删除原模板工作表
        wb.remove(template_sheet)

