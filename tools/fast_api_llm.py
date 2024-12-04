from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import uvicorn
from argparse import ArgumentParser
import torch


def _load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16,  device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path)
  
    return model, tokenizer

app = FastAPI()

@app.post("/qwen-api/")
async def process(request: Request):
    param = await request.json()
    query = param.get('message', None)
    if query is None:
        return {'message': 'Invalid message.'}

    input_ids = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("LLM: ", response)
    return {'message': response, 'len_token': len(generated_ids[0])}


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default='path to weights',
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9005)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()
    model, tokenizer = _load_model(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")