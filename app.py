import os
import gradio as gr
import json
import pickle
from neo4j import GraphDatabase
import logging
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
import re
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# ロギングの設定
logging.basicConfig(level=logging.INFO)

# Neo4jHandlerクラス
class Neo4jHandler:
    def __init__(self, uri, username, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.session = self.driver.session()
            logging.info("Neo4jへの接続に成功しました。")
        except Exception as e:
            logging.error(f"Neo4jへの接続に失敗しました。エラーの詳細: {e}")
            self.driver = None
            self.session = None

    def close(self):
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        if self.session:
            try:
                result = self.session.run(query, parameters)
                return result
            except Exception as e:
                logging.error(f"クエリ実行中にエラーが発生しました: {e}")
                return None
        else:
            logging.error("セッションが開いていません。")
            return None

# エンティティ抽出モデルの定義
class Entities(BaseModel):
    names: List[str] = Field(..., description="All the person, organization, or business entities that appear in the text")

# エンティティ抽出関数の定義
entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Extract all person and organization entities from the following text and return them in JSON format as a list under the key 'names'.\n\nText: {question}")
])

def extract_entities(llm, question: str) -> Entities:
    formatted_prompt = entity_prompt.format_messages(question=question)
    response = llm(formatted_prompt)
    try:
        entities = Entities.parse_raw(response.content)
    except json.JSONDecodeError:
        logging.error(f"LLMのレスポンスをJSONとしてパースできませんでした。レスポンス内容: {response.content}")
        entities = Entities(names=[])
    return entities

def remove_lucene_chars(input_str):
    lucene_special_chars = r'[+\-&|!(){}[\]^"~*?:\\/]'
    return re.sub(lucene_special_chars, ' ', input_str)

def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    if words:
        full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(llm, neo4j_handler, question: str):
    result = ""
    entities = extract_entities(llm, question)
    for entity in entities.names:
        query = """
        CALL db.index.fulltext.queryNodes('entity', $query, {limit:20})
        YIELD node, score
        CALL {
          WITH node
          MATCH (node)-[r]->(neighbor)
          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
          UNION ALL
          WITH node
          MATCH (node)<-[r]-(neighbor)
          RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
        }
        RETURN output LIMIT 1000
        """
        parameters = {"query": generate_full_text_query(entity)}
        response = neo4j_handler.run_query(query, parameters)
        if response:
            for record in response:
                result += record["output"] + "\n"
    return result.strip()

# LangChainのチェーン設定
template = """あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。文脈から考えられる回答の正確度、信頼度が70%を下回ったら「不確実な情報が含まれている場合があります。」と回答してください。
{context}
ユーザーの質問: {question}"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def setup_chain(llm):
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain

def load_embeddings(file):
    try:
        if isinstance(file, str):  # ファイルパスが渡された場合
            with open(file, 'rb') as f:
                graph_documents = pickle.load(f)
        else:  # Gradioのファイルオブジェクトが渡された場合
            graph_documents = pickle.loads(file.read())

        # GraphDocumentをDocumentに変換
        documents = []
        for graph_doc in graph_documents:
            content = ""
            if hasattr(graph_doc, 'nodes'):
                content += "Nodes:\n"
                for node in graph_doc.nodes:
                    content += f"{node}\n"
            if hasattr(graph_doc, 'relationships'):
                content += "Relationships:\n"
                for rel in graph_doc.relationships:
                    content += f"{rel}\n"
            doc = Document(page_content=content)
            documents.append(doc)
        return documents
    except Exception as e:
        logging.error(f"PKLファイルの読み込み中にエラーが発生しました: {e}")
        return None

def initialize_components(openai_api_key, neo4j_uri, neo4j_username, neo4j_password, pkl_file=None):
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
        
        neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)
        if neo4j_handler.session is None:
            return None, "Neo4jへの接続に失敗しました。"
        
        chain = setup_chain(llm)
        
        vectorstore = None
        if pkl_file is not None:
            documents = load_embeddings(pkl_file)
            if documents is None:
                return None, "PKLファイルの読み込みに失敗しました。"
            
            # エンベディングモデルの作成
            embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # ベクトルストアの作成
            vectorstore = FAISS.from_documents(documents, embeddings_model)
        
        components = {
            "llm": llm,
            "chain": chain,
            "neo4j_handler": neo4j_handler,
            "vectorstore": vectorstore
        }
        return components, "Neo4jとベクトルストアの初期化に成功しました。"
    
    except Exception as e:
        logging.error(f"初期化中にエラーが発生しました: {e}")
        return None, f"初期化中にエラーが発生しました: {e}"

def chatbot_interface(components, question, history):
    llm = components.get("llm")
    chain = components.get("chain")
    neo4j_handler = components.get("neo4j_handler")
    vectorstore = components.get("vectorstore")
    
    if llm is None or neo4j_handler is None or vectorstore is None:
        return history + [("エラー", "セットアップが完了していません。")]
    try:
        # Structured data retrieval
        structured_data = structured_retriever(llm, neo4j_handler, question)
        
        # ベクトル検索で関連文書を取得
        docs = vectorstore.similarity_search(question, k=3)
        retrieved_docs = "\n".join([doc.page_content for doc in docs])
        
        # コンテキストの作成
        final_context = f"Structured data:\n{structured_data}\nRetrieved documents:\n{retrieved_docs}"
        
        response = chain.run({"context": final_context, "question": question})
        return history + [(question, response)]
    except Exception as e:
        logging.error(f"チャットボット処理中にエラーが発生しました: {e}")
        return history + [(question, f"エラーが発生しました: {e}")]

def get_pkl_files():
    pkl_folder = "PKL"
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    pkl_files = [f for f in os.listdir(pkl_folder) if f.endswith('.pkl')]
    return pkl_files

# Gradio インターフェース
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chat Application")
    
    state = gr.State()
    chat_history = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            openai_api_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
            gr.Markdown("[OpenAI APIキーの取得はこちら](https://platform.openai.com/api-keys)")
            neo4j_uri_input = gr.Textbox(label="Neo4j URI", value="neo4j+s://3887c0d9.databases.neo4j.io")
            neo4j_username_input = gr.Textbox(label="Neo4j Username", value="neo4j")
            neo4j_password_input = gr.Textbox(label="Neo4j Password", type="password")
            pkl_file_select = gr.Dropdown(label="PKLファイルを選択", choices=get_pkl_files())
            setup_button = gr.Button("Setup")
        with gr.Column(scale=2):
            status_display = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(label="質問を入力してください", scale=4)
        send_button = gr.Button("送信", scale=1)
    
    gr.Markdown("### テンプレート質問")
    gr.Markdown("1. 日本の繊維産業にはいくつの産地がありますか？それらの名前を挙げてください。")
    gr.Markdown("2. 混紡糸の一種として挙げられている製品は何ですか？")
    gr.Markdown("3. 日本の繊維産業の歴史や伝統について、何か情報はありますか？")
    
    clear = gr.Button("会話をリフレッシュ")
    save_log = gr.Button("会話ログを保存")

    def handle_setup(api_key, uri, username, password, pkl_file):
        pkl_path = os.path.join("PKL", pkl_file) if pkl_file else None
        components, status = initialize_components(api_key, uri, username, password, pkl_path)
        return components, status

    def handle_chat(components, question, history):
        if components is None:
            return history, history, "セットアップが完了していません。"
        new_history = chatbot_interface(components, question, history)
        return new_history, new_history, "回答生成完了"

    def handle_clear():
        return [], []

    def handle_save_log(history):
        log_content = "\n".join([f"Q: {q}\nA: {a}\n" for q, a in history])
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as temp_file:
            temp_file.write(log_content)
        return temp_file.name

    setup_button.click(handle_setup, 
                       inputs=[openai_api_input, neo4j_uri_input, neo4j_username_input, neo4j_password_input, pkl_file_select], 
                       outputs=[state, status_display])
    send_button.click(handle_chat, inputs=[state, msg, chat_history], outputs=[chatbot, chat_history, status_display])
    msg.submit(handle_chat, inputs=[state, msg, chat_history], outputs=[chatbot, chat_history, status_display])
    clear.click(handle_clear, outputs=[chatbot, chat_history])
    save_log.click(handle_save_log, inputs=[chat_history], outputs=[gr.File(label="ダウンロード")])

# アプリケーションの起動
if __name__ == "__main__":
    demo.launch()
