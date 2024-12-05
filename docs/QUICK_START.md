## Quick Start
The communication between Scene Understanding Module (VLM, Qwen-VL), Analytic/Heuristic Process (LLM, GPT-4/Qwen1.5) and CARLA is based on FastAPI.
## Step-1: Launch VLM and LLM on the server
* Download the pretrained checkpoints of [Qwen-VL-7B](https://huggingface.co/Jianbiao/qwenvl-7b-scene-understanding) and [Qwen1.5-1.8B](https://huggingface.co/jianbiao/qwen1.5-decision).
* Run the Program.
```
conda activate Qwen-VL
python tools/fast_api_vlm -c [path to weights] --port 9000
conda activate Qwen1.5
python tools/fast_api_llm -c [path to weights] --port 9005
```

## Step-2: Launch CARLA simulator locally
```
cd [YOUR ROOT TO CARLA]
./CarlaUE4.sh --world-port=${carla_port} --resX=800 --resY=600 -quality-level=low
```
## Step-3: Port Mapping
If all modules such as VLM and LLM are running locally or can be accessed directly via the public internet, you can skip this step.

Map the server-side service ports to your local machine:
```
# Assume you can connect to server as: ssh username@server_adress -p server_port
ssh -N username@server_adress -p server_port -L 9000:localhost:vlm_port -L 9005:localhost:llm_port

```


## Step-4: Launch Agent
### Configuration

* Before running the script, ensure you have your OpenAI `api_key` and `proxies` set up. Then, modify the variants in the [config.py](../team_code/config.py).
```
api_key = "" # your openai api_key 
proxies = {
    "https": "" # your proxies
}
```
* We provide some accumulated samples in the [memory database](../memory/). Feel free to use them! 
```
memory_data_path = "./memory/test.json"
memory_embedding_path = "./memory/test.npy"
memory_database_path = "./memory/test.db"
```

* If you want to utilize Analytic Process (GPT-4) for decision making, modify the variant `LIGHT_LLM` in the [config.py](../team_code/config.py) to False.

### Run the script
```
./leaderboard/scripts/eval_leapad.sh ${carla_port} ${traffic_port} ${vlm_port} ${llm_port}
```

