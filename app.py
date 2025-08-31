import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# =========================
# 設定
# =========================
st.set_page_config(page_title="LLM Expert Web App", page_icon="🤖", layout="centered")

# OpenAI APIキーの取得（Streamlit Cloud では、[Settings] → [Secrets] に OPENAI_API_KEY を入れておく）
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が見つかりません。環境変数または Streamlit Secrets に設定してください。")
    st.stop()

# =========================
# 専門家モード（A / B）
# =========================
EXPERTS = {
    "A": {
        "label": "A: データサイエンティスト",
        "system": (
            "あなたはトップクラスのデータサイエンティストです。"
            "統計・機械学習・実験設計・可視化に精通し、数式や具体例を用いて、"
            "再現可能で実務に活かせるアドバイスを簡潔に提示してください。"
        ),
    },
    "B": {
        "label": "B: UXデザイナー",
        "system": (
            "あなたは熟練のUXデザイナーです。"
            "ユーザー課題の洞察、情報設計、ペルソナ/ユーザーフロー、プロトタイピング、"
            "可用性テストに基づく改善提案を、実装可能な粒度で分かりやすく提示してください。"
        ),
    },
}

# =========================
# LLM 呼び出し関数（要件の関数）
#  - 入力: input_text（入力テキスト）, expert_key（ラジオボタン選択値 'A' or 'B'）
#  - 出力: 文字列（LLMからの回答）
# =========================
def ask_llm(input_text: str, expert_key: str) -> str:
    """入力テキストと専門家モード(A/B)を受け、LLMの回答を返す。"""
    if expert_key not in EXPERTS:
        raise ValueError("Unknown expert key")

    system_prompt = EXPERTS[expert_key]["system"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",   # 必要に応じて変更可（例: "gpt-4o"）
        temperature=0.2,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": input_text})

# =========================
# UI
# =========================
st.title("🤖 LLM Expert Web App")
st.caption("LangChain + Streamlit で作るシンプルな LLM アプリ")

with st.expander("このアプリについて / 使い方", expanded=True):
    st.markdown(
        """
**概要**  
- テキスト入力フォームに質問・相談内容を記入し、**専門家モード（A/B）**を選択して送信すると、  
  選んだ専門家としての視点・言葉遣いで LLM が回答します。  
- 仕組み：LangChain を用い、**選択されたモードに応じた system メッセージ**を組み立てて LLM に渡しています。

**使い方**  
1. **専門家モード**を選びます（A=データサイエンティスト / B=UXデザイナー）。  
2. 入力欄にテキストを記入します（例：課題や要件、制約、現状データなど）。  
3. **送信**を押すと、画面下部に回答が表示されます。
        """
    )

# ラジオボタン（表示はラベル、値は 'A' / 'B'）
expert_choice = st.radio(
    "専門家モードを選択してください：",
    options=list(EXPERTS.keys()),
    format_func=lambda k: EXPERTS[k]["label"],
    horizontal=True,
)

# 入力フォーム
with st.form("qa_form", clear_on_submit=False):
    user_text = st.text_area(
        "入力テキスト（質問・相談内容）",
        placeholder="例：A/Bテストの設計で悩んでいます。分割や必要サンプルサイズ、指標はどう決めるべき？",
        height=150,
    )
    submitted = st.form_submit_button("送信する")

if submitted:
    if not user_text.strip():
        st.warning("入力テキストを入力してください。")
    else:
        with st.spinner("LLM からの回答を生成中…"):
            try:
                answer = ask_llm(user_text, expert_choice)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
            else:
                st.subheader("🧠 回答")
                st.write(answer)