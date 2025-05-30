import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv(".env.local", override=True)


# ---------------------------------------------------------------------------
# グラフ生成関数
# ---------------------------------------------------------------------------
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def devide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        return "0 で割ることはできません"
    return a / b


async def build_graph():
    """エージェントを生成し、Supervisor グラフをコンパイルして返す"""

    print("--- LangGraph Supervisor グラフを構築中 ---")

    with Path("mcp_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    brave_key = os.getenv("BRAVE_API_KEY")
    if not brave_key:
        raise RuntimeError("環境変数 BRAVE_API_KEY が設定されていません")
    config["mcpServers"]["brave-search"]["env"]["BRAVE_API_KEY"] = brave_key
    client = MultiServerMCPClient(config["mcpServers"])

    # MCP サーバーが公開しているツール仕様を取得
    mcp_tools = await client.get_tools()

    # ユーティリティエージェント
    travel_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        tools=mcp_tools,
        name="travel_agent",
        prompt="あなたは，Braveによる観光地検索と，現地時間の正確な取得，服装の提案を行う旅行コーディネートエージェントです。",
    )

    print("--- 旅行コーディネートエージェントを完了 ---")
    # 掛け算専門エージェント -------------------------------------------
    print("--- 数学アシスタントエージェントを構築中 ---")
    math_assistant = create_react_agent(
        model="openai:gpt-4.1-mini",
        tools=[multiply, devide],
        name="math_assistant",
        prompt=(
            "あなたは数学の専門家です。ユーザーからの計算リクエストに応じて、"
            "掛け算や割り算を行い、正確な結果を返してください。"
        ),
    )
    print("--- 数学アシスタントエージェントを完了 ---")
    # Supervisor グラフをコンパイル -----------------------------------
    print("--- Supervisor グラフを構築中 ---")
    supervisor_prompt = (
        "あなたはオーケストレーターです。次の 2 つのサブエージェントを管理します:\n"
        "1. travel_agent – MCP 経由で旅行のコーディネートを担当\n"
        "2. math_assistant – 難しい計算を正確に実行\n"
        "ユーザーのリクエストを読み取り、どのエージェントにタスクを割り当てるか判断し、"
        "結果を収集してユーザーにわかりやすく返答してください。"
    )

    supervisor = create_supervisor(
        agents=[travel_agent, math_assistant],
        model=ChatOpenAI(model="gpt-4.1-mini"),
        prompt=supervisor_prompt,
    ).compile()

    print("--- Supervisor グラフの構築が完了 ---")

    return supervisor


# ---------------------------------------------------------------------------
# デモ実行関数
# ---------------------------------------------------------------------------


async def interactive_demo() -> None:
    supervisor = await build_graph()

    print("質問を入力してください（空行 or Ctrl-C で終了）\n")
    while True:
        try:
            user_query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_query:
            print("Bye!")
            break

        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_query}]}
        ):
            print(chunk)
            print("\n")


if __name__ == "__main__":
    asyncio.run(interactive_demo())
