import subprocess
import sys

from fastapi import FastAPI

from skynet import http_client
from skynet.env import (
    disable_llm_health_check,
    llama_n_ctx,
    llama_path,
    openai_api_base_url,
    openai_api_port,
    use_oci,
    use_vllm,
)
from skynet.logs import get_logger
from skynet.modules.ttt.openai_api.slim_router import router as slim_router

log = get_logger(__name__)

app = FastAPI()
app.include_router(slim_router)


def initialize():
    if not use_vllm:
        return

    log.info(f'Starting vLLM server on port {openai_api_port} using model {llama_path}')

    proc = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'vllm.entrypoints.openai.api_server',
            '--disable-log-requests',
            '--model',
            llama_path,
            '--gpu_memory_utilization',
            str(0.90),
            '--max-model-len',
            str(llama_n_ctx),
            '--port',
            str(openai_api_port),
        ],
        shell=False,
    )

    if proc.poll() is not None:
        log.error('Failed to start vLLM OpenAI API server')
    else:
        log.info('vLLM OpenAI API server started')


async def is_ready():
    if use_oci or disable_llm_health_check:
        return True

    # /v1/models works for any OpenAI-compatible upstream (remote vLLM, LiteLLM, Ollama).
    url = f'{openai_api_base_url}/health' if use_vllm else f'{openai_api_base_url}/v1/models'

    try:
        response = await http_client.request('GET', url)
        response.release()
        return response.status == 200
    except Exception:
        return False


__all__ = ['app', 'initialize', 'is_ready']
