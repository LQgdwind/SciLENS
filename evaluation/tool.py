from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from typing import AsyncIterator

from openai_harmony import (
    Author,
    Role,
    Message,
    TextContent,
)

def _maybe_update_inplace_and_validate_channel(
    *, input_message: Message, tool_message: Message
) -> None:

    if tool_message.channel != input_message.channel:
        if tool_message.channel is None:
            tool_message.channel = input_message.channel
        else:
            raise ValueError(
                f"Messages from tool should have the same channel ({tool_message.channel=}) as "
                f"the triggering message ({input_message.channel=})."
            )

class Tool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:

    @property
    def output_channel_should_match_input_channel(self) -> bool:
        return True

    async def process(self, message: Message) -> AsyncIterator[Message]:
        async for m in self._process(message):
            if self.output_channel_should_match_input_channel:
                _maybe_update_inplace_and_validate_channel(input_message=message, tool_message=m)
            yield m

    @abstractmethod
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        if False:
            yield
        _ = message
        raise NotImplementedError

    @abstractmethod
    def instruction(self) -> str:
        raise NotImplementedError

    def instruction_dict(self) -> dict[str, str]:
        return {self.name: self.instruction()}

    def error_message(
        self, error_message: str, id: UUID | None = None, channel: str | None = None
    ) -> Message:
        return Message(
            id=id if id else uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=TextContent(text=error_message),
            channel=channel,
        ).with_recipient("assistant")
