from fastapi import FastAPI 
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel 

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

import uuid

model_path = "/home/user_id/models/Meta-Llama-3.1-8B-Instruct/"

engine_args = AsyncEngineArgs(
    model=model_path, 
    gpu_memory_utilization=0.95, 
    tensor_parallel_size=1
)
llm = AsyncLLMEngine.from_engine_args(engine_args)

sampling_params = SamplingParams(
    temperature=0.5, 
    top_p=0.7, 
    repetition_penalty=1.1, 
    max_tokens=1024
)

app = FastAPI() 

class QueryRequest(BaseModel):
    query: str

@app.post("/generate/")
async def generate_post(request: QueryRequest):
    request_id = str(uuid.uuid4())
    sent_text = "" 

    async def stream_response(): 
        nonlocal sent_text
        results_generator = llm.generate(request.query, sampling_params, request_id=request_id)

        async for output in results_generator:
            text = output.outputs[0].text

            new_text = text[len(sent_text):]
            sent_text = text

            for word in new_text.split(" "):
                if word:  
                    yield word + " "

            if output.finished:
                return

    return StreamingResponse(stream_response(), media_type="text/plain")