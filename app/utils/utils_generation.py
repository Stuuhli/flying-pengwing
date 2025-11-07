from app.config import BACKEND_FASTAPI_LOG
from app.utils.utils_logging import initialize_logging, logger
from app.utils.utils_LLM_process_inputs import Refine
from copy import deepcopy
from llama_index.core import Settings
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import messages_to_prompt as default_messages_to_prompt
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ContentBlock, TextBlock
from llama_index.core.base.query_pipeline.query import QueryComponent
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate, BasePromptTemplate
from llama_index.core.prompts.base import PromptComponent
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.prompt_utils import get_biggest_prompt
from llama_index.core.prompts.utils import get_template_vars, format_string
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.types import BasePydanticProgram, RESPONSE_TEXT_TYPE, BaseOutputParser
from typing import Callable, Dict, List, Optional, Type, Any, Sequence, Union, Tuple
initialize_logging(BACKEND_FASTAPI_LOG)

def get_prefix_messages_with_context(
    context_template: PromptTemplate,
    system_prompt: str,
    prefix_messages: List[ChatMessage],
    chat_history: List[ChatMessage],
    llm_metadata_system_role: MessageRole,
) -> List[ChatMessage]:
    context_str_w_sys_prompt = system_prompt.strip() + context_template.template 
    return [
        ChatMessage(content=context_str_w_sys_prompt, role=llm_metadata_system_role),
        *prefix_messages,
        *chat_history,
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]


def get_response_synthesizer(
    llm: Optional[LLM] = None,
    prompt_helper: Optional[PromptHelper] = None,
    text_qa_template: Optional[BasePromptTemplate] = None,
    refine_template: Optional[BasePromptTemplate] = None,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    callback_manager: Optional[CallbackManager] = None,
    use_async: bool = False,
    streaming: bool = False,
    structured_answer_filtering: bool = False,
    output_cls: Optional[Type[BaseModel]] = None,
    program_factory: Optional[
        Callable[[BasePromptTemplate], BasePydanticProgram]
    ] = None,
    verbose: bool = False,
) -> BaseSynthesizer:
    """Get a response synthesizer."""
    text_qa_template = text_qa_template
    refine_template = refine_template

    callback_manager = callback_manager or Settings.callback_manager
    llm = llm or Settings.llm
    prompt_helper = (
        prompt_helper
        or Settings._prompt_helper
        or PromptHelper.from_llm_metadata(
            llm.metadata,
        )
    )

    if response_mode == ResponseMode.REFINE:
        return Refine(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            output_cls=output_cls,
            streaming=streaming,
            structured_answer_filtering=structured_answer_filtering,
            program_factory=program_factory,
            verbose=verbose,
        )
    elif response_mode == ResponseMode.COMPACT:
        return CompactAndRefine(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            output_cls=output_cls,
            streaming=streaming,
            structured_answer_filtering=structured_answer_filtering,
            program_factory=program_factory,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown mode: {response_mode}")
    
class CompactAndRefine(Refine):
    """Refine responses across compact text chunks."""

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        logger.debug("Context before compaction: %s \n and length of list: %s", text_chunks, len(text_chunks))
        compact_texts = self._make_compact_text_chunks(query_str, text_chunks)
        logger.info("Context after compaction: %s \n and length of list: %s", compact_texts, len(compact_texts))
        return await super().aget_response(
            query_str=query_str,
            text_chunks=compact_texts,
            prev_response=prev_response,
            **response_kwargs,
        )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        # TODO: This is a temporary fix - reason it's temporary is that
        # the refine template does not account for size of previous answer.
        logger.debug("Context before compaction: %s \n and length of list: %s", text_chunks, len(text_chunks))
        new_texts = self._make_compact_text_chunks(query_str, text_chunks)
        logger.info("Context after compaction: %s \n and length of list: %s", new_texts, len(new_texts))
        return super().get_response(
            query_str=query_str,
            text_chunks=new_texts,
            prev_response=prev_response,
            **response_kwargs,
        )

    def _make_compact_text_chunks(
        self, query_str: str, text_chunks: Sequence[str]
    ) -> List[str]:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = get_biggest_prompt([text_qa_template, refine_template])
        return self._prompt_helper.repack(max_prompt, text_chunks, llm=self._llm)
    

class ChatPromptTemplate(BasePromptTemplate):
    message_templates: List[ChatMessage]

    def __init__(
        self,
        message_templates: Sequence[ChatMessage],
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseOutputParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ):
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = []
        for message_template in message_templates:
            template_vars.extend(get_template_vars(message_template.content or ""))

        super().__init__(
            message_templates=message_templates,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_vars=template_vars,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    @classmethod
    def from_messages(
        cls,
        message_templates: Union[List[Tuple[str, str]], List[ChatMessage]],
        **kwargs: Any,
    ) -> "ChatPromptTemplate":
        """From messages."""
        if isinstance(message_templates[0], tuple):
            message_templates = [
                ChatMessage.from_str(role=role, content=content)  # type: ignore[arg-type]
                for role, content in message_templates
            ]
        return cls(message_templates=message_templates, **kwargs)  # type: ignore[arg-type]

    def partial_format(self, **kwargs: Any) -> "ChatPromptTemplate":
        prompt = deepcopy(self)
        prompt.kwargs.update(kwargs)
        return prompt

    def format(
        self,
        llm: Optional[BaseLLM] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        **kwargs: Any,
    ) -> str:
        del llm  # unused
        messages = self.format_messages(**kwargs)

        if messages_to_prompt is not None:
            return messages_to_prompt(messages)

        return default_messages_to_prompt(messages)

    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        del llm  # unused
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        mapped_all_kwargs = self._map_all_vars(all_kwargs)

        messages: List[ChatMessage] = []
        for message_template in self.message_templates:
            # Handle messages with multiple blocks
            if message_template.blocks:
                formatted_blocks: List[ContentBlock] = []
                for block in message_template.blocks:
                    if isinstance(block, TextBlock):
                        template_vars = get_template_vars(block.text)
                        relevant_kwargs = {
                            k: v
                            for k, v in mapped_all_kwargs.items()
                            if k in template_vars
                        }
                        formatted_text = format_string(block.text, **relevant_kwargs)
                        formatted_blocks.append(TextBlock(text=formatted_text))
                    else:
                        # For non-text blocks (like images), keep them as is
                        # TODO: can images be formatted as variables?
                        formatted_blocks.append(block)

                message = message_template.model_copy()
                message.blocks = formatted_blocks
                messages.append(message)
            else:
                # Handle empty messages (if any)
                messages.append(message_template.model_copy())

        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)
        logger.info("Message to LLM before predict %s", messages)
        return messages

    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        return default_messages_to_prompt(self.message_templates)

    def _as_query_component(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> QueryComponent:
        """As query component."""
        return PromptComponent(prompt=self, format_messages=True, llm=llm)