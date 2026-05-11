import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from oci.exceptions import TransientServiceError
from openai.types.chat import ChatCompletionMessageParam

from skynet.env import oci_blackout_fallback_duration, use_oci
from skynet.logs import get_logger
from skynet.modules.monitoring import MAP_REDUCE_CHUNKING_COUNTER
from skynet.modules.ttt.llm_selector import LLMSelector
from skynet.modules.ttt.ratelimit_tracker import (
    extract_ratelimit_from_response,
    get_ratelimit_callback,
    should_track_ratelimit,
)
from skynet.modules.ttt.summaries.prompts.action_items import (
    action_items_conversation,
    action_items_emails,
    action_items_meeting,
    action_items_text,
)
from skynet.modules.ttt.summaries.prompts.summary import (
    summary_conversation,
    summary_emails,
    summary_meeting,
    summary_text,
)
from skynet.modules.ttt.summaries.prompts.table_of_contents import (
    table_of_contents_conversation,
    table_of_contents_emails,
    table_of_contents_meeting,
    table_of_contents_text,
)
from skynet.modules.ttt.summaries.v1.models import DocumentPayload, HintType, Job, JobType, Processors

log = get_logger(__name__)

# Global OCI blackout state management
_oci_blackout_until: Optional[datetime] = None


def set_oci_blackout(duration_seconds: int) -> None:
    """Set OCI blackout for the specified duration."""
    global _oci_blackout_until
    _oci_blackout_until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
    log.warning(f"OCI blackout set until {_oci_blackout_until} ({duration_seconds} seconds)")


def is_oci_blackout_active() -> bool:
    """Check if OCI is currently in blackout period."""
    global _oci_blackout_until
    if _oci_blackout_until is None:
        return False

    now = datetime.now(timezone.utc)
    if now >= _oci_blackout_until:
        _oci_blackout_until = None  # Clear expired blackout
        log.info("OCI blackout period expired, resuming normal processing")
        return False

    return True


hint_type_to_prompt = {
    JobType.SUMMARY: {
        HintType.CONVERSATION: summary_conversation,
        HintType.EMAILS: summary_emails,
        HintType.MEETING: summary_meeting,
        HintType.TEXT: summary_text,
    },
    JobType.ACTION_ITEMS: {
        HintType.CONVERSATION: action_items_conversation,
        HintType.EMAILS: action_items_emails,
        HintType.MEETING: action_items_meeting,
        HintType.TEXT: action_items_text,
    },
    JobType.TABLE_OF_CONTENTS: {
        HintType.CONVERSATION: table_of_contents_conversation,
        HintType.EMAILS: table_of_contents_emails,
        HintType.MEETING: table_of_contents_meeting,
        HintType.TEXT: table_of_contents_text,
    },
}


async def summarize(model: BaseChatModel, job: Job) -> str:
    payload = job.payload
    job_type = job.type
    customer_id = job.metadata.customer_id
    text = payload.text

    # Fallback priority: payload.prompt -> customer's live_summary_prompt (if is_live_summary=True) -> hint_type_to_prompt[job_type][payload.hint]
    system_message = payload.prompt

    if not system_message and payload.is_live_summary:
        from skynet.modules.ttt.customer_configs.utils import get_existing_customer_config

        config = await get_existing_customer_config(customer_id)
        if config:
            system_message = config.get('live_summary_prompt')

    if not system_message:
        prompt_fn = hint_type_to_prompt[job_type][payload.hint]
        system_message = prompt_fn(payload.preferred_locale)

    prompt = ChatPromptTemplate(
        [
            ('system', system_message),
            ('human', '{text}'),
        ]
    )

    # Build LCEL chain
    chain = prompt | model | StrOutputParser()

    # Add rate limit callback for system's own API key
    callbacks = []
    if should_track_ratelimit(customer_id):
        processor = LLMSelector.get_job_processor(customer_id)
        callbacks.append(get_ratelimit_callback(processor.value))

    config = {'callbacks': callbacks} if callbacks else {}

    async def invoke_with_retry(input_text: str, context: str) -> str:
        """Invoke chain with retry on empty result."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            result = await chain.ainvoke({'text': input_text}, config=config)
            if result and result.strip():
                if attempt > 0:
                    log.info(f'job {job.id} succeeded on {context} after {attempt + 1} attempts')
                return result
            if attempt < max_retries:
                log.info(
                    f'job {job.id} got empty result on {context} (attempt {attempt + 1}/{max_retries + 1}), retrying...'
                )
            else:
                log.info(f'job {job.id} got empty result on {context} after {max_retries + 1} attempts')
        return result

    # Estimate tokens to decide if we need map-reduce
    num_tokens = model.get_num_tokens(text)
    context_window = LLMSelector.get_context_window(customer_id)
    threshold = context_window * 3 / 4

    if num_tokens < threshold:
        # Simple case: text fits in context window
        result = await invoke_with_retry(text, 'simple summarize')
    else:
        # Map-reduce: split, summarize chunks in parallel, then combine
        num_chunks = int(num_tokens // threshold + 1)
        chunk_size = int(num_tokens // num_chunks)

        log.info(f'Splitting text into {num_chunks} chunks of {chunk_size} tokens')
        MAP_REDUCE_CHUNKING_COUNTER.labels(job_type=job_type.value).inc()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=100)
        docs = text_splitter.create_documents([text])

        # Map: summarize each chunk in parallel
        chunk_tasks = [chain.ainvoke({'text': doc.page_content}, config=config) for doc in docs]
        chunk_summaries = await asyncio.gather(*chunk_tasks)

        # Reduce: combine summaries
        combined_text = '\n\n'.join(chunk_summaries)
        result = await invoke_with_retry(combined_text, 'map-reduce combine')

    formatted_result = result.strip()

    log.info(f'job {job.id} input length: {len(system_message) + len(text)}')
    log.info(f'job {job.id} output length: {len(formatted_result)}')

    return formatted_result


async def process_text(model: BaseChatModel, payload: DocumentPayload) -> str:
    prompt = payload.prompt.strip()
    text = payload.text.strip()

    prompt_template = ChatPromptTemplate(
        [
            ('system', prompt),
            ('human', '{text}'),
        ]
    )

    chain = prompt_template | model | StrOutputParser()
    result = await chain.ainvoke(input={'text': text})

    log.info(f'input length: {len(prompt) + len(text)}')
    log.info(f'output length: {len(result)}')

    return result


async def process(job: Job) -> str:
    payload = job.payload
    job_type = job.type
    customer_id = job.metadata.customer_id

    llm = LLMSelector.select(
        customer_id,
        job_id=job.id,
        oci_blackout=is_oci_blackout_active(),
        **{'max_completion_tokens': payload.max_completion_tokens},
    )

    try:
        if job_type in [JobType.SUMMARY, JobType.ACTION_ITEMS, JobType.TABLE_OF_CONTENTS]:
            result = await summarize(llm, job)
        elif job_type == JobType.PROCESS_TEXT:
            result = await process_text(llm, payload)
        else:
            raise ValueError(f'Invalid job type {job_type}')
    except TransientServiceError as e:
        log.warning(f"Job {job.id} hit TransientServiceError: {e}")

        # Set blackout using fallback duration
        blackout_duration = oci_blackout_fallback_duration
        log.info(f"TransientServiceError detected, setting {blackout_duration}s blackout")
        set_oci_blackout(blackout_duration)

        # Switch current job to local processing
        LLMSelector.override_job_processor(job.id, Processors.LOCAL)
        return await process(job)

    except Exception as e:
        log.warning(f"Job {job.id} failed: {e}")

        processor = LLMSelector.get_job_processor(customer_id, job.id)

        if processor == Processors.OCI and not use_oci:
            LLMSelector.override_job_processor(job.id, Processors.LOCAL)
            return await process(job)

        raise e

    return result


async def process_chat_completion(
    messages: List[ChatCompletionMessageParam], customer_id: Optional[str] = None, **model_kwargs
) -> str:
    llm = LLMSelector.select(customer_id, **model_kwargs)

    response = await llm.ainvoke(messages)

    # Track rate limits for system's own API key
    if customer_id and should_track_ratelimit(customer_id):
        processor = LLMSelector.get_job_processor(customer_id)
        extract_ratelimit_from_response(response, processor.value)

    return response.content


async def process_chat_completion_stream(
    messages: List[ChatCompletionMessageParam], customer_id: Optional[str] = None, **model_kwargs
):
    llm = LLMSelector.select(customer_id, **model_kwargs)
    track_ratelimit = customer_id and should_track_ratelimit(customer_id)
    first_chunk = True

    try:
        async for chunk in llm.astream(messages):
            # Track rate limits from first chunk (headers only available there)
            if first_chunk and track_ratelimit:
                processor = LLMSelector.get_job_processor(customer_id)
                extract_ratelimit_from_response(chunk, processor.value)
                first_chunk = False

            yield chunk.content if hasattr(chunk, 'content') else str(chunk)
    except Exception as e:
        yield json.dumps(
            {'error': e.body if hasattr(e, 'body') else str(e), 'code': e.code if hasattr(e, 'code') else None}
        ) + '\n'
