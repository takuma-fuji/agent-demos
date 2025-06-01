"""A terminal chatbot that helps elementary‑school students discover a **summer‑holiday research topic**.

Powered by the **OpenAI Agents SDK**.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool
from agents.extensions.handoff_prompt import (
    prompt_with_handoff_instructions as add_handoff_info,
)

# -----------------------------------------------------------------------------
# 0. Load environment variables (optional – for local dev)
# -----------------------------------------------------------------------------
load_dotenv(".env.local", override=True)

# -----------------------------------------------------------------------------
# 1. Shared tool – Web Search (available to specialist agents)
# -----------------------------------------------------------------------------
web_search_tool = WebSearchTool()

# -----------------------------------------------------------------------------
# 2. Specialist agents (each receives *full* conversation history via handoff)
#    ※ handoffs は **生成時** に渡す
# -----------------------------------------------------------------------------

question_deepener_agent = Agent(
    name="問い深掘りエージェント",
    handoff_description="選ばれた問いを具体的なサブ問いと調べ方に分解する専門家",
    instructions=add_handoff_info(
        "あなたは研究計画を一緒に立てるメンターです。与えられたメイン問いを基に、より詳しく調べられる **3つのサブ問い** を挙げ、各サブ問いについて調べ方や実験アイデアを具体的に提案してください。\n\n"
        "最後に研究計画のまとめを箇条書きで示し、大学生にもわかりやすい手順と準備物を説明してください。そのうえで『exit』または『quit』と入力すれば終了できると伝えてください。"
    ),
    tools=[web_search_tool],
    handoffs=[],  # チェーンの終端
)

question_decider_agent = Agent(
    name="問い決定エージェント",
    handoff_description="興味を基に中心となる研究の問いを作る専門家",
    instructions=add_handoff_info(
        "あなたは卒業研究のテーマを提案する先生です。渡された興味リストを読み取り、\n"
        "ネット検索でサーベイをして、興味に関連する面白そうなテーマ案を列挙してください。\n"
        "学生との会話を繰り返して、テーマ案のうち最も興味を持ってもらえそうなものを選び、\n"
        "『○○はどうして△△なのだろう？』の形で **1つ** のメイン問いを提案し、その問いを選んだ理由も短く添えてください。\n\n"
        "提案が完了したら handoff ツール **`transfer_to_問い深掘りエージェント`** を呼び出してください。"
    ),
    tools=[web_search_tool],
    handoffs=[question_deepener_agent],
)

interest_finder_agent = Agent(
    name="興味探しエージェント",
    handoff_description="学生の興味や好きを聞き出す専門家",
    instructions=add_handoff_info(
        "あなたは大学生への聞き取りが得意です。質問と学生の回答を繰り返しながら、学生の興味や好きなことを最大3個のキーワードにまとめてください。\n"
        "興味がまとまったら handoff ツール **`transfer_to_問い決定エージェント`** を呼び出してください。"
        "学生の興味をJSON形式で **`transfer_to_問い決定エージェント`** に渡してください（例: {\"interests\": [\"故障診断\", \"看護ロボット\", \"経路最適化\"]}）。\n\n"
        
    ),
    handoffs=[question_decider_agent],
)

self_intro_agent = Agent(
    name="自己紹介エージェント",
    handoff_description="自分の役割を大学生にわかりやすく紹介する専門家",
    instructions=add_handoff_info(
        "あなたは大学4年の卒業研究サポートボットです。最初に丁寧な自己紹介を行い、これから一緒に研究テーマを見つける流れを簡潔に説明してください。\n\n"
        "自己紹介が終わったら handoff ツール **`transfer_to_興味探しエージェント`** を呼び出してください。"
    ),
    handoffs=[interest_finder_agent],
)

# -----------------------------------------------------------------------------
# 3. Orchestrator – kicks off the flow only **once** per session
# -----------------------------------------------------------------------------

orchestrator_agent = Agent(
    name="卒業研究コンダクター",
    instructions=add_handoff_info(
        "あなたは大学4年の卒業研究課題を決めるための総合エージェントです。\n"
        "会話開始時、まだ自己紹介が行われていない場合は **`transfer_to_自己紹介エージェント`** を呼び出し、以降はエージェント間の handoff に任せてください。"
    ),
    handoffs=[self_intro_agent],
)

# -----------------------------------------------------------------------------
# 4. Terminal REPL following the *conversation loop* pattern
# -----------------------------------------------------------------------------

WELCOME_MESSAGE = (
    "----- 🌻 大学4年卒業研究チャットボットへようこそ！ -----\n"
    "(終了するときは 'exit' または 'quit' と入力してください)\n"
)


def main() -> None:
    """Run an interactive chat session that maintains conversation state across turns."""

    print(WELCOME_MESSAGE)
    print("ボット: こんにちは！私は大学4年の卒業研究を一緒に考えるお手伝いをします。まずは自己紹介してください。")

    # Conversation state
    conversation_items: list = []  # accumulated list of ResponseInputItem
    current_agent: Agent = orchestrator_agent  # start at orchestrator

    while True:
        user_str = input("あなた: ").strip()
        if user_str.lower() in {"exit", "quit"}:
            print("ボット: ありがとうございました！楽しい卒業研究にしてくださいね。👋")
            break

        # Append the new user message in Responses format
        conversation_items.append({"role": "user", "content": user_str})

        # Run the agents with the *full* conversation so far
        result = Runner.run_sync(current_agent, conversation_items)

        # Show final output to the user
        print("ボット:", result.final_output)

        # 🔄 Update state for the next turn
        conversation_items = result.to_input_list()          # includes everything so far
        current_agent = result.last_agent                   # resume where we left off


if __name__ == "__main__":
    main()
