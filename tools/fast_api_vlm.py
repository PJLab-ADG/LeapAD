from fastapi import FastAPI, Request
import base64
from PIL import Image
import io
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import uvicorn
from argparse import ArgumentParser
import time


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default='path to weights',
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-d", "--tmp-dir", type=str, default='tmp')
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)

    args = parser.parse_args()
    return args


def _load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if 'lora' in args.checkpoint_path:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map="auto",
            trust_remote_code=True,
            resume_download=True,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            args.checkpoint_path, trust_remote_code=True, resume_download=True,
        )

    return model, tokenizer


app = FastAPI()
args = _get_args()

MAX_LEN=42
VLM_SYSTEM_PROMPT_TEMPLATE= '''Current scene: {}, <img>{}</img>. You are driving a car in an urban street and this image indicates the scene that you see. Please describe in detail the key objects in the scene that may affect your driving. The object information should contain the object's category, the object's location relative to the ego car or the ego lane, the object's motion state, and the object's bounding box, as well as the approximate distance from the ego car. Especially, if there are any traffic lights, the information should contain the traffic light's color and bounding box.'''
@app.post("/qwenvl-api/")
async def process(request: Request):
    param = await request.json()
    base64_image = param.get('base64_image', None)
    # base64_image = param.get('focus_base64', None)

    camera_view = param.get('camera_view', 'CAM_FRONT')
    if base64_image is None:
        return {'message': 'Invalid image.'}

    base64_image_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(base64_image_bytes)).convert('RGB')

    file_names = sorted(os.listdir(f'./{args.tmp_dir}'))
    if len(file_names) > MAX_LEN:
        os.remove(os.path.join(f'./{args.tmp_dir}', file_names[0]))
    tmp_name = f'./{args.tmp_dir}/{time.time()}.jpg'
    img.save(tmp_name)

    message = VLM_SYSTEM_PROMPT_TEMPLATE.format(camera_view, tmp_name)
    response, history = model.chat(tokenizer, message, history=[])

    print("VLM: ", response)
    return {'message': response}


if __name__ == '__main__':
    os.makedirs(args.tmp_dir, exist_ok=True)
    model, tokenizer = _load_model(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")