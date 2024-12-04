api_key = "" # your openai api_key 
proxies = {
    "https": "" # your proxies
}
headers = {
"Content-Type": "application/json",
"Authorization": f"Bearer {api_key}"
}

CAMERA_YAWS = [-60, 0, 60]
INIT_SPEED = 5 # initial speed
STOP_TIME = 40 # STOP_TIME/LLM_FREQUENCE second
LLM_FREQENCE = 2 # decision frequence
SELECT_OFFSET_BEFORE = 2 # selected waypoint offset. (several points are duplicated with waypoint (1 before, 1 after))
SELECT_OFFSET_AFTER = 2
LLM_FOCUS_FACTOR = 4
DST = 10
GPT_ON = True
HIGH_SPEED = 8.3
FAST_SIM = True
LIGHT_LLM = True
FEW_SHOT = 3
REFLECTION_SAMPLES = 10  # samples for reflection
SAMPLE_INTER = 2  # sample interval for reflection

# sensor config
full_fov = 60
image_width = 1600 # 
image_height = 1200
image_show_width = 800
image_show_height = 600
focus_scale_h = 4.
focus_scale_w = 3.
pitch = 10 # deg
show_box = False


memory_data_path = "./memory/test.json"
memory_embedding_path = "./memory/test.npy"
memory_database_path = "./memory/test.db"

samples_per = 1
collection_name = "test-runtime"