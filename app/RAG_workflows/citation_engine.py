"""
Citation workflow engine which converts the k retreivals into k llamaindex nodes 
Then creates citation format followed by generation
"""

from typing import List, Union
from llama_index.core.base.llms.types import MessageRole,ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.workflow import Event
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.data_structs import Node
from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
)
from llama_index.core.response_synthesizers import (
    ResponseMode,  # noqa: F811
)
from app.config import BACKEND_FASTAPI_LOG, response_artificial
from app.prompt_config import CITATION_QA_TEMPLATE, CITATION_REFINE_TEMPLATE
from app.utils.utils_logging import initialize_logging, logger
from app.utils.utils_generation import get_prefix_messages_with_context, get_response_synthesizer, ChatPromptTemplate
initialize_logging(BACKEND_FASTAPI_LOG)

""" Citation Engine """
class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class CreateCitationsEvent(Event):
    """Add citations to the nodes."""

    nodes: list[NodeWithScore] 


DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 40

class CitationQueryEngineWorkflow(Workflow):
    def __init__(self, LLM: LLM, memory: BaseMemory, prefix_messages: List[ChatMessage]= [], timeout = 180, disable_validation = False, verbose = False, service_manager = None, num_concurrent_runs = None, system_prompt= None):
        super().__init__(timeout, disable_validation, verbose, service_manager, num_concurrent_runs)
        self.system_prompt= system_prompt
        self._LLM= LLM
        self.memory= memory
        self._prefix_messages = prefix_messages
        self.context_template_citation= CITATION_QA_TEMPLATE
        self.refine_context_template_citation= CITATION_REFINE_TEMPLATE 
    
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> Union[RetrieverEvent, None]:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        #print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        retrievals= ev.get("results")
        if retrievals is None:
            print("No documents were retreivedd")
            return None
        nodes= []
        for retrieval in retrievals:
            node= NodeWithScore(node= Node(text=retrieval["Document"], id_= retrieval["Doc_id"]), score=retrieval["score"])
            nodes.append(node)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def create_citation_nodes(
        self, ev: RetrieverEvent
    ) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        nodes = ev.nodes

        new_nodes: List[NodeWithScore] = []

        # text_splitter = SentenceSplitter(
        #     chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
        #     chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        # )

        for node in nodes:
            text= node.text
            new_text= f"Source {len(new_nodes)+1}:\n{text}\n"
            new_base_node = TextNode.model_validate(node.node.dict())
            new_node = NodeWithScore(
                    node=new_base_node, score=node.score
                )
            new_node.node.set_content(new_text)
            new_nodes.append(new_node)

        print("Created citations.")
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a streaming response using the retrieved nodes."""
        query = await ctx.get("query", default=None)
        chat_history = self.memory.get(
            input=query,
        )
        chat_history= chat_history[-2:]
        # Get the messages for the QA and refine prompts
        qa_messages = get_prefix_messages_with_context(
            CITATION_QA_TEMPLATE,
            system_prompt=str(self.system_prompt),
            prefix_messages=self._prefix_messages,
            chat_history=chat_history,
            llm_metadata_system_role=self._LLM.metadata.system_role,
        )
        refine_messages = get_prefix_messages_with_context(
            CITATION_REFINE_TEMPLATE,
            system_prompt=str(self.system_prompt),
            prefix_messages=self._prefix_messages,
            chat_history=chat_history,
            llm_metadata_system_role=self._LLM.metadata.system_role,
        )
        
        qa_messages_chatprompt= ChatPromptTemplate.from_messages(qa_messages)
        logger.info("after chat template, qa_messages: %s", qa_messages_chatprompt)
        refine_messages_chatprompt= ChatPromptTemplate.from_messages(refine_messages)
        logger.info("after chat template, refine_messages: %s", refine_messages_chatprompt)
        synthesizer = get_response_synthesizer(
            llm=self._LLM,
            text_qa_template=qa_messages_chatprompt,
            refine_template=refine_messages_chatprompt,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
            verbose=True
        )
        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        #response= response_artificial
        user_message = ChatMessage(content=query, role=MessageRole.USER)
        ai_message = ChatMessage(content=str(response), role=MessageRole.ASSISTANT)
        self.memory.put(user_message)
        self.memory.put(ai_message)

        return StopEvent(result=response)

