from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv(".env.local", override=True)

mcp = FastMCP("rag-retriever")
client = OpenAI()


@mcp.tool()
async def clothing_suggestion(query: str) -> str:
    """
    服装の提案を行うツール
    :param query: ユーザーからの服装に関する質問
    :return: 服装の提案
    """
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "あなたは服装の提案を行う専門家です。"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    mcp.run(transport="stdio")
