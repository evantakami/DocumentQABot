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
    
def process_excel_like_data(df):
    """
    针对一个无表头的 DataFrame：
      1. 查找包含「項番」和「作業概要」的表头行；
      2. 前向填充该行作为新列名，并取其下方数据；
      3. 删除「項番」与「作業概要」之间原始表头为空的列；
      4. 将同一大标题下的多列合并为单列；
      5. 以「項番」向下填充后按項番分组合并多行；
    返回整理好的 DataFrame。
    """
    hdr_row = find_header_row(df, keywords=("項番", "作業概要"), max_search=10)
    if hdr_row is None:
        print("在前 10 行内未找到同时包含『項番』『作業概要』的行，无法继续。")
        return None

    # 取出原始表头（不做填充），用于判断哪些列原本为空
    original_header = df.iloc[hdr_row]
    # 前向填充，用于后续作为列名
    ffilled_header = original_header.ffill()
    df_data = df.iloc[hdr_row+1:].copy()
    df_data.columns = ffilled_header

    # 新增：删除「項番」和「作業概要」之间原始表头为空的列
    start_index = None
    end_index = None
    # 遍历 ffill 后的表头，找到「項番」和「作業概要」的位置
    for i, col in enumerate(ffilled_header):
        if col == "項番" and start_index is None:
            start_index = i
        if col == "作業概要" and start_index is not None and end_index is None:
            end_index = i
    # 如果找到了两个位置，并且二者之间有列
    if start_index is not None and end_index is not None and end_index > start_index:
        indices_to_drop = []
        # 遍历这两个位置之间的原始表头
        for i in range(start_index+1, end_index):
            orig_val = original_header.iloc[i]
            # 如果原始表头为空（或仅为空白字符串），则记录该列索引
            if pd.isna(orig_val) or str(orig_val).strip() == "":
                indices_to_drop.append(i)
        # 按列索引删除这些列（注意：删除操作不会影响其他部分的列）
        df_data.drop(df_data.columns[indices_to_drop], axis=1, inplace=True)

    # 以下代码保持原有逻辑，对同一大标题下的多列进行合并
    unique_titles = []
    for col in df_data.columns:
        if col not in unique_titles:
            unique_titles.append(col)

    title_to_indices = {}
    col_list = list(df_data.columns)
    for i, t in enumerate(col_list):
        title_to_indices.setdefault(t, []).append(i)

    merged_dict = {}
    for t in unique_titles:
        indices = title_to_indices[t]
        merged_dict[t] = df_data.apply(lambda row: merge_columns_in_row(row, indices, t), axis=1)
    df_merged_cols = pd.DataFrame(merged_dict)

    if "項番" not in df_merged_cols.columns:
        print("合并后未发现「項番」列，无法进行行合并。")
        return df_merged_cols

    df_merged_cols["項番"] = df_merged_cols["項番"].ffill()
    df_final = df_merged_cols.groupby("項番", as_index=False).apply(merge_rows_in_group)
    df_final.reset_index(drop=True, inplace=True)

    return df_final

