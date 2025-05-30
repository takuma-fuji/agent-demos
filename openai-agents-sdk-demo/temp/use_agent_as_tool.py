# summer_research_chatbot.py
"""A terminal chatbot that helps elementary school students find a summer‑holiday research topic.

Built with the OpenAI Agents SDK.

Usage:
  1.  Ensure you have Python 3.9+ and an OpenAI API key.
      $ export OPENAI_API_KEY="sk‑..."
  2.  Install dependencies inside a virtual environment:
      $ python -m venv .venv && source .venv/bin/activate
      $ pip install openai-agents
  3.  Run the bot:
      $ python summer_research_chatbot.py

Type "exit" to quit the session.
"""

from __future__ import annotations

import asyncio
import os
from typing import List

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

from agents import Agent, Runner, WebSearchTool

# ---------------------------------------------------------------------------
# Hosted tool (web search) — lets downstream agents ground their suggestions
# ---------------------------------------------------------------------------
web_search_tool = WebSearchTool()

# ---------------------------------------------------------------------------
# Specialist agents (each one becomes a *tool* for the orchestrator agent)
# ---------------------------------------------------------------------------

self_intro_agent = Agent(
    name="自己紹介エージェント",
    handoff_description="自分の役割を小学生にわかりやすく紹介する専門家",
    instructions=(
        "あなたは夏休みの自由研究サポートボットです。最初に丁寧な自己紹介をして、"
        "これから一緒に研究テーマを見つけることを簡潔に説明してください。"
    ),
)

interest_finder_agent = Agent(
    name="興味探しエージェント",
    handoff_description="子どもの興味や好きを聞き出す専門家",
    instructions=(
        "あなたは小学生への聞き取りが得意です。質問をしながら生徒の興味や好きなことを最大3個のキーワードにまとめ、"
        "必ず JSON 形式で返してください（例: {\"interests\": [\"昆虫\", \"宇宙\", \"水の実験\"]}）。"
    ),
)

question_decider_agent = Agent(
    name="問い決定エージェント",
    handoff_description="興味を基に中心となる研究の問いを作る専門家",
    instructions=(
        "あなたは自由研究のテーマを決める先生です。渡された興味リストを読み取り、"
        "『○○はどうして△△なのだろう？』の形で1つのメインとなる問いを提案してください。"
        "その際、問いを選んだ簡単な理由も添えてください。"
    ),
    tools=[web_search_tool],
)

question_deepener_agent = Agent(
    name="問い深掘りエージェント",
    handoff_description="選ばれた問いを具体的なサブ問いと調べ方に分解する専門家",
    instructions=(
        "あなたは研究計画を一緒に立てるメンターです。与えられたメインの問いを読み取り、"
        "より詳しく調べられる3つのサブ問いを挙げ、各サブ問いについて調べ方や実験のアイデアを具体的に提案してください。"
    ),
    tools=[web_search_tool],
)

# ---------------------------------------------------------------------------
# Orchestrator agent – drives the conversation, calling the above agents/tools
# ---------------------------------------------------------------------------

orchestrator_agent = Agent(
    name="自由研究コンダクター",
    instructions=(
        "あなたは夏休みの自由研究課題を決めるための総合エージェントです。"
        "会話の流れに合わせて以下のツールを順番に使い、小学生と対話しながら進めてください。\n"
        "各ステップが終わってから次のステップに進むようにしてください。\n\n"
        "1. 最初の自己紹介: 使うべきツールは，self_introduction\n"
        "2. 生徒の興味を聞き出して整理: 使うべきツールは，find_interests\n"
        "3. 興味に基づいてメインとなる問いを提案: 使うべきツールは，decide_question\n"
        "4. サブ問いと調査計画を提案: 使うべきツールは，deepen_question\n\n"
        "ツール呼び出し結果を生徒にわかりやすい日本語で説明し、必要に応じて追加の質問をして理解を深めましょう。"
        "最後に、全てのステップが完了したら、研究計画をまとめて生徒に伝えてください。\n"
        "そして，'exit' または 'quit'での終了を促してください．"
    ),
    tools=[
        self_intro_agent.as_tool(
            tool_name="self_introduction",
            tool_description="小学生に向けてわかりやすい自己紹介を行う",
        ),
        interest_finder_agent.as_tool(
            tool_name="find_interests",
            tool_description="生徒の興味や好きなことを質問し、整理する",
        ),
        question_decider_agent.as_tool(
            tool_name="decide_question",
            tool_description="興味リストをもとに中心となる研究の問いを決める",
        ),
        question_deepener_agent.as_tool(
            tool_name="deepen_question",
            tool_description="メインの問いを深掘りし、サブ問いと調べ方を提案する",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Simple REPL for terminal use
# ---------------------------------------------------------------------------

WELCOME_MESSAGE = "----- 🌻 夏休み自由研究チャットボットへようこそ！ -----\n(終了するときは 'exit' または 'quit' と入力してください)\n"


def main() -> None:
    """Run an interactive chat session synchronously (blocking)."""

    print(WELCOME_MESSAGE)
    while True:
        user_input = input("あなた: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("ボット: ありがとうございました！楽しい自由研究にしてくださいね。👋")
            break

        # Run the orchestrator agent synchronously so we can stay inside the REPL loop.
        result = Runner.run_sync(orchestrator_agent, input=user_input)
        print("ボット:", result.final_output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
