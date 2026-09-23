"""A scripted chat model: replays AIMessages in order, so agent graphs run offline."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field


class ScriptedModel(GenericFakeChatModel):
    offered: list[list[str]] = Field(default_factory=list)   # tool names per model request

    def bind_tools(self, tools, **kwargs):   # agents bind tools; the script already knows them
        self.offered.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self


def scripted(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies), disable_streaming=True)


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def calls(*tool_calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(tool_calls))


def say(text: str) -> AIMessage:
    return AIMessage(content=text)
