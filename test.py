import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISABLE_DEVICES"] = "0"


# 모델 위치
# model_path = "/home/xxxjjhhh/my_model/Meta-Llama-3.1-8B-Instruct/"
model_path = "/mnt/disks/vllm/llama3/Llama-3.1-8B-Instruct/"

def main():
    llm = LLM(model=model_path, enforce_eager=True, gpu_memory_utilization=0.9, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.5, top_p=0.7, repetition_penalty=1.1, max_tokens=1024)

    query = "해리포터의 줄거리를 한글로 간략히 설명해 주세요."
    response = llm.generate(query, sampling_params)

    print(response[0].outputs[0].text)

if __name__ == '__main__':
    main()
