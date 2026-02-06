import os
import uvicorn
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 

from vllm import LLM, SamplingParams

os.environ["CUDA_VISABLE_DEVICES"] = "0"

app = FastAPI() 

model_path = "/home/user_id/models/Meta-Llama-3.1-8B-Instruct/"
llm = LLM(model=model_path, gpu_memory_utilization=0.9,  tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.5, top_p=0.7, repetition_penalty=1.1, max_tokens=1024)


# 서버 설정
class QueryRequest(BaseModel):
    query: str

@app.post("/generate/")
async def generate_response(request: QueryRequest):
    try:
        response = llm.generate(request.query, sampling_params)
        result_text = response[0].outputs[0].text
        return {"response": result_text}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))