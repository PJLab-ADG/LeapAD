#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import math
import time
import json
from threading import Thread
import cv2
import numpy as np
import io
from PIL import Image
import base64
import requests
import re
import matplotlib.colors as mcolors
import random
import os
import chromadb


from waypointer import Waypointer
from pid import PIDController
from prompts import SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from config import *
from utils import *
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.timer import GameTime
import datetime
from collections import deque
import copy

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.utils.route_manipulation import downsample_route



VLM_PORT = os.environ.get('VLM_PORT', 9000)
LLM_PORT = os.environ.get('LLM_port', 9005)
HUMAN_AGENT_NAME = os.environ.get('HUMAN_AGENT', 'LeanAD Agent')

exist_ok = os.path.exists(memory_database_path)
client = chromadb.PersistentClient(path=memory_database_path)
if not exist_ok:
    collection = client.create_collection(collection_name)
    with open(memory_data_path) as f:  
        memory_data = json.load(f)
    memory_embeddings = np.load(memory_embedding_path)
    messages = []
    idxs = []
    embeddings = []
    for idx, message in enumerate(memory_data):
        if idx%samples_per !=0:
            continue
        messages.append({'messages': json.dumps(message['messages'])})
        idxs.append(str(idx))
        embeddings.append(memory_embeddings[idx].tolist())
    collection.add(
        embeddings=embeddings,
        metadatas=messages,
        ids=idxs
    )
else:
    collection = client.get_collection(collection_name)


def RoadOption2Str(cmd):
    if cmd == RoadOption.LEFT:
        return 'LEFT'
    elif cmd == RoadOption.RIGHT:
        return 'RIGHT'
    elif cmd == RoadOption.CHANGELANELEFT:  
        return 'CHANGELANELEFT'
    elif cmd == RoadOption.CHANGELANERIGHT:
        return 'CHANGELANERIGHT'
    elif cmd == RoadOption.STRAIGHT:
        return 'STRAIGHT'
    else:  # LANEFOLLOW
        return 'LANEFOLLOW'


def aggregate_message(scene_description, ego_state, notice_info, history_message=None):
    if LIGHT_LLM:
        text = "## Scene Description\n The important object information from the front view: \n" + scene_description['front_view'] + '\n' + "The traffic light information from the far-sighted view: \n" + scene_description['focus_view'] + '\n' + f"## Ego state\n Your current speed is {ego_state['speed']} m/s. Your current steering value is {ego_state['steer']}.\n"
        if 'turn' in notice_info:
            text += f"## Notice\n You are turning. You should drive at a low speed (< 3 m/s) to deal with emergencies and should not pay attention to any vehicles(>5m) and traffic lights since the traffic lights can't affect your action.\n"
        if 'stop' in notice_info:
            text += f"## Notice\n You have stopped for a long time, maybe you shold check out the surroundings and move on to complete the road.\n"
        if 'cross' in notice_info:
            text += f"## Notice\n You are approaching the traffic intersection, you should drive at a low speed (< 3 m/s) to deal with emergencies. You should pay more attention on the traffic lights. If there exist traffic light in both front view and far-sighted view, you should obey the rule of the traffic lights in the far-sighted view.\n"
        if 'stop sign' in notice_info:
            text += f"## Notice\n There is a stop sign in the front, you should stop!"
        message = []
        message.append({
            'role': 'user',
            'content': text
        })
    else:
        message=[]
        message.append({
            "type":"text",
            "text": SYSTEM_PROMPT
        })
        message.append({
            "type":"text",
            "text": "## Scene Description\n The important object information from the front view: \n" + scene_description['front_view'] + '\n' + "The traffic light information from the far-sighted view: \n" + scene_description['focus_view'] + '\n'
        })
        message.append({
            "type":"text",
            "text": f"## Ego state\n Your current speed is {ego_state['speed']} m/s. Your current steering value is {ego_state['steer']}.\n",
        })
        if 'turn' in notice_info:
            message.append({
                "type":"text",
                "text": f"## Notice\n You are turning and you should not pay attention to any vehicles(>5m) traffic lights since the traffic lights can't affect your action.\n",
            })
        if 'stop' in notice_info:
            message.append({
                "type":"text",
                "text": f"## Notice\n You have stopped for a long time, maybe you shold check out the surroundings and move on to complete the road.\n",
            })
        if 'cross' in notice_info:
            message.append({
                "type":"text",
                "text": f"## Notice\n You are approaching the traffic intersection, you should drive at a low speed (< 3 m/s) to deal with emergencies. You should pay more attention on the traffic lights. If there exist traffic light in both front view and far-sighted view, you should obey the rule of the traffic lights in the far-sighted view.\n",
            })
        if 'stop sign' in notice_info:
            message.append({
                "type":"text",
                "text": f"## Notice\n There is a stop sign less than 5m away in the front, you should stop!",
            })
    return message


def request_openai(content, max_tokens: int = 4000, model_type='gpt-4o'):
    payload = {
        "model": model_type,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, proxies=proxies)

    return response


def process_img(img):
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((image_show_width, image_show_height))
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    img_byte = img_byte.getvalue()
    image_base64 = base64.b64encode(img_byte).decode('utf-8')
    return image_base64


def process_img2(img):
    img_pil = img.resize((image_show_width, image_show_height))
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    img_byte = img_byte.getvalue()
    image_base64 = base64.b64encode(img_byte).decode('utf-8')
    return image_base64


def fetch_response_with_try_loop(image_base64=None, message=None):
    if image_base64 is not None:
        flag = True
        while flag:
            try:
                response = requests.post("http://localhost:{}/qwenvl-api".format(VLM_PORT), json={"base64_image":image_base64}).json()["message"]
                flag = False
            except:
                print('Warning! Qwen-VL reqeust error, try again!')

    if message is not None:
        flag = True
        while flag:
            try:
                response = request_openai(message).choices[0].message.content if not LIGHT_LLM else requests.post("http://localhost:{}/qwen-api".format(LLM_PORT), json={"message":message}).json()["message"]
                flag = False
            except:
                print(f'Warning! {"GPT" if not LIGHT_LLM else "Qwen"} reqeust error, try again!')
    return response


def fetch_embeddings_with_try_loop(text):
    flag = True
    payload = {
                "model": 'text-embedding-3-small',
                "input": text
                }
    while flag:
        
        try:
            response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload,proxies=proxies)
            response.raise_for_status()  # Raise an exception for bad status codes
            embeddings = response.json()['data'][0]['embedding']
            flag = False  
        except:
            print('Warning! Openai request error, try again!')
            time.sleep(10)
    return embeddings


def fetch_gpt_reflection_with_try_loop(event_message, history_info):
    message=[]
    message.append({
        "type":"text",
        "text": REFLECTION_SYSTEM_PROMPT
    })
    message.append({
        "type":"text",
        "text": f"## Current Event Message\n {event_message}."
    })
    message.append({
        "type":"text",
        "text": f'''## Historical Descriptions and Decisions\n {["(time="+x['time']+ "," + str(x['messages'])+")" for x in history_info]}'''
    })
    
    flag = True
    count_request = 0
    while flag:
        count_request += 1
        if count_request >= 10:
            print('Request timeout!')
            break
        try:
            response = request_openai(message).json()['choices'][0]['message']['content']
            flag = False
        except:
            print(f'Warning! GPT reqeust error, try again!')
            
    scene_time = re.findall(r'## Time\ntime=\d+', response)
    if len(scene_time) == 0:
            scene_time = re.findall(r'## Time\n\d+', response)

    reflected_answer = re.findall(r'## Reasoning.*?## Decision\n\w+', response, re.DOTALL)
    ta_dict = {}
    for t, a in zip(scene_time, reflected_answer):
        ta_dict[t.replace('## Time\ntime=', '')] = a
    reflected_messages, reflected_embeddings = [], []

    for x in history_info:
        if x['time'] in ta_dict.keys():
            reflected_message = copy.deepcopy(x['messages'])
            reflected_message[-1]['content'] = ta_dict[x['time']]
            reflected_messages.append({'messages': json.dumps(reflected_message)})
            reflected_embeddings.append(x['embedding'])
    
    return reflected_messages, reflected_embeddings


def update_desired_speed(ans, desired_speed, ac_factor=1.0, dc_factor=1):
    match = re.search(r'## Decision\n(.*)', ans)
    if match:
        decision = match.group(1)
        if decision == 'AC':
            desired_speed += ac_factor
        elif decision == 'DC':
            desired_speed -= dc_factor
        elif decision == 'IDLE':
            pass
        elif decision == 'STOP':
            desired_speed = 0
        else:
            pass
        desired_speed = max(0, desired_speed)
        desired_speed = min(desired_speed, HIGH_SPEED)

    return desired_speed


def get_entry_point():
    return 'LeapadAgent'


class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self):
        self._width = 800
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("{}".format(HUMAN_AGENT_NAME))

    def run_interface(self, input_data, speed, control, focus_bbox=None, bbox=None, num_frame=None, save_flag=False, unique_id='id', Delta_x=0, Delta_y=0, reflection_flag=False, accident_flag=False, crop_img=None, nowTime=None, stop_flag=None):
        """
        Run the GUI
        """

        fisheye_left = np.zeros((600, 800, 3), np.uint8)
        fisheye_left[:,:800] = input_data['fisheye_left'][1][:,:,-2::-1]

        fisheye_right = np.zeros((600, 800, 3), np.uint8)
        fisheye_right[:,:800] = input_data['fisheye_right'][1][:,:,-2::-1]


        new_size = (int(image_width/focus_scale_w), int(image_height/focus_scale_h)) 
        resized_img = crop_img.resize(new_size)  

        surface_focus_crop = np.array(resized_img)
        
        surface = np.zeros((600, 800, 3), np.uint8)
        full_image = cv2.resize(input_data['Center'][1][:, :, -2::-1], (800, 600))

        surface[:, :800] = full_image # (600, 800, 3)
        
        # display image
        if save_flag and reflection_flag or accident_flag and not stop_flag: 
            cv2.imwrite(f'replay/{unique_id}/{nowTime}_focus.png', surface_focus_crop[..., ::-1])
            cv2.imwrite(f'replay/{unique_id}/{nowTime}.png', surface[..., ::-1])  
            cv2.imwrite(f'replay/{unique_id}/{nowTime}_left.png', fisheye_left[..., ::-1])
            cv2.imwrite(f'replay/{unique_id}/{nowTime}_right.png', fisheye_right[..., ::-1])

        # Text
        speed_str = "Speed: "+"{:.2f}".format(speed) + " m/s"
        surface = cv2.putText(surface, speed_str, (40, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        control_str = "Throttle: "+"{:.2f}".format(control.throttle) + " steer: "+"{:.2f}".format(
            control.steer) + " brake: " + "{:.1f}".format(control.brake)
        surface = cv2.putText(surface, control_str, (250, 560),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if bbox != None and show_box:
            for box in bbox:
                color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()]) # init color
                rgb_color = mcolors.to_rgb(mcolors.TABLEAU_COLORS[color])  
                h, w = surface.shape[:2]
                surface = cv2.rectangle(surface, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), rgb_color, 2)

        if focus_bbox != None and show_box:
            for box in focus_bbox:
                color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()]) # init color
                rgb_color = mcolors.to_rgb(mcolors.TABLEAU_COLORS[color])  
                h, w = surface_focus_crop.shape[:2]
                surface_focus_crop = cv2.rectangle(surface_focus_crop, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), rgb_color, 2)

        surface_focus_crop = cv2.resize(surface_focus_crop, (200, 150))
        surface[:150,300:500] = surface_focus_crop

        self._surface = pygame.surfarray.make_surface(
            surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


class LeapadAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False
    unique_id = time.time()
    num_frame = 0
    save_flag = False
    stop_flag = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        os.makedirs(f"replay/{self.unique_id}", exist_ok=True)
        self.record_file = open(f"replay/{self.unique_id}/log.txt", 'a')
        self.track = Track.SENSORS

        self.agent_engaged = False
        self._hic = HumanInterface()
        # Config
        self.camera_x = 1.5
        self.camera_z = 2.4
        self.turn_KP = 1.0
        self.turn_KI = 0.5
        self.turn_KD = 0.2
        self.turn_n = 20  # buffer size
        self.speed_KP = 5.0
        self.speed_KI = 0.5
        self.speed_KD = 1.0
        self.speed_n = 40  # buffer size
        self.pre_speed = 0.


        self.waypointer = None
        self.turn_controller = PIDController(
            K_P=self.turn_KP, K_I=self.turn_KI, K_D=self.turn_KD, n=self.turn_n)
        self.speed_controller = PIDController(
            K_P=self.speed_KP, K_I=self.speed_KI, K_D=self.speed_KD, n=self.speed_n)

        self.record_time = 0
        self.desired_speed = INIT_SPEED
        self.steer = 0
        self.bbox = None
        self.focus_bbox = None
        self.focus_idx = -1
        self.stop_time = self.num_frame
        self.odom_info = {
            'last_cxy': None,
            'waypoint': None,
            'next_wxy': None
        }
        self.event_messages = None
        self.queue_for_reflection = deque()
        self.stop_sign_flag = False
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

    def rotate_point(self, pointx, pointy, yaw_rad):
        new_pointx = pointx * math.cos(yaw_rad) + pointy * math.sin(yaw_rad)
        new_pointy = -pointx * math.sin(yaw_rad) + pointy * math.cos(yaw_rad)

        return new_pointx, new_pointy

    def pid_control(self, aim, speed, desired_speed):
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = steer if abs(steer) > 0.05 else 0.0

        brake = desired_speed < 0.8
        if desired_speed < 0.2:
            brake = 1.0
            throttle = 0.0
        elif desired_speed < 0.9 * speed:
            delta = np.clip(speed - desired_speed, 0.0, 5.0)
            brake = delta / 5.0
            brake = np.clip(brake, 0.0, 1.0)
            throttle = 0.0
        else:
            delta = np.clip(desired_speed - speed, 0.0, 0.5)
            throttle = self.speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.8)
            brake = 0.0

        return float(steer), float(throttle), float(brake)

    def draw_waypoints(self, world, gps):
        for idx, w in enumerate(self._global_plan_world_coord):
            wp = carla.Location(w[0].location.x, w[0].location.y, 1.0)

            size = 0.1
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            if idx <= self.waypointer.current_idx:
                continue

            world.debug.draw_point(wp, size=size, color=color, life_time=0.1)

        lat, lon, _ = gps
        cur_x, cur_y = self.waypointer.latlon_to_xy(lat, lon)
        cur_x, cur_y = cur_y, -cur_x

        wp = carla.Location(cur_x, cur_y, 1.0)
        world.debug.draw_point(
            wp, size=0.1, color=carla.Color(255, 0, 0), life_time=0.1)
        return

    def sensors(self):
        sensors = [
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0.,
                'y': 0., 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.other.imu',  'x': 0., 'y': 0., 'z': self.camera_z,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'IMU'},

        ]

        # Add cameras
        sensors.append(
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': image_width, 'height': image_height, 'fov': full_fov, 'id': 'Center'},
        )
        sensors.append(
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': -1.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -105,
             'width': 800, 'height': 600, 'fov': 150, 'id': 'fisheye_left'},
        )
        sensors.append(
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 1.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 105,
             'width': 800, 'height': 600, 'fov': 150, 'id': 'fisheye_right'},
        )

        return sensors

    def __call__(self, world=None, event_messages=None):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()
        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()

        control = self.run_step(input_data, timestamp, world, event_messages)
        control.manual_gear_shift = False

        return control

    def run_step(self, input_data, timestamp, world=None, event_messages=None):
        """
        Execute one step of navigation.
        """   
        reflection_flag = ((self.event_messages is not None and len(event_messages) > len(self.event_messages)) or (self.event_messages is None and len(event_messages)>0)) and REFLECTION_SAMPLES > 0 and LIGHT_LLM
        accident_flag = ((self.event_messages is not None and len(event_messages) > len(self.event_messages)) or (self.event_messages is None and len(event_messages)>0))
        if reflection_flag:
            print(event_messages[-1])
            accident_list = []
            json_path = f"replay/{self.unique_id}/accident.json"
            if os.path.exists(json_path):
                with open(json_path) as f:
                    accident_list = json.load(f)
            else:
                accident_list = []

            history_info = list(self.queue_for_reflection) # keys: ["time", "messages", "embedding"]
            accident_list.append({'event_message':event_messages[-1], 'reflection_samples':[{'time': x['time'], 'messages': x['messages']} for x in history_info]})
            with open(json_path, 'w') as f:
                json.dump(accident_list, f, indent=4)

            reflected_messages, reflected_embeddings = fetch_gpt_reflection_with_try_loop(event_messages[-1], history_info[::SAMPLE_INTER]) # sample inter
            
            collection.add(
                embeddings=reflected_embeddings,
                metadatas=reflected_messages,
                ids=[f'{time.time()}_r{i}' for i in range(len(reflected_messages))]
            )
            
        self.event_messages = event_messages
        self.agent_engaged = True

        # sensor information
        _, gps = input_data.get('GPS')
        _, imu = input_data.get('IMU')
        _, ego = input_data.get('EGO')
        spd = ego.get('speed')

        # waypoint
        if self.waypointer is None:
            self.waypointer = Waypointer(
                self._global_plan, gps
            )
        self.waypointer.tick(gps)
        if not GPT_ON:
            self.draw_waypoints(world, gps)

        lat, lon, _ = gps
        cur_x, cur_y = self.waypointer.latlon_to_xy(lat, lon)

        # distance from the candidate waypoint
        for i in range(SELECT_OFFSET_BEFORE, SELECT_OFFSET_BEFORE+7, 1):
            candidate_idx = self.waypointer.current_idx + i

            if candidate_idx < len(self.waypointer.global_plan):
                cx, cy, cmd= self.waypointer.global_plan[candidate_idx-1]
                
                if cmd in [RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT] and np.linalg.norm([cur_x-cx, cur_y-cy])<DST:
                    if self.focus_idx == -1:
                        self.focus_idx = candidate_idx
                        self.odom_info['waypoint'] = (cx, cy)
                        self.odom_info['next_wxy'] = self.waypointer.global_plan[candidate_idx][:2]
                    break

        if self.focus_idx > 0 and self.waypointer.current_idx >= SELECT_OFFSET_AFTER + self.focus_idx:
            self.focus_idx = 0
        if self.focus_idx == 0 and self.waypointer.global_plan[self.waypointer.current_idx][2] not in [RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]:
                self.focus_idx = -1

        if self.waypointer.current_idx >= self.focus_idx+2:
            self.odom_info['waypoint'] = None
            self.odom_info['last_wxy'] = None

        # compute waypoint
        Delta_x = 0
        Delta_y = -pitch / (full_fov/2.) * (image_height / 2.)
        if self.odom_info['waypoint'] is not None:
            cross_a = compute_angle_using_two_points(self.odom_info['next_wxy'], self.odom_info['waypoint'])
            Delta_a = ((cross_a-imu[6] + np.pi)% (2 * np.pi) - np.pi)/math.pi*180.0 # x
            Delta_x = int(Delta_a / (full_fov/2.) * (image_width / 2.))
            
            if Delta_x+image_width/2.0 + image_width/focus_scale_w/2. > image_width:
                Delta_x = int(image_width/2. - image_width/focus_scale_w/2.)-1.
            if Delta_x+image_width/2.0 - image_width/focus_scale_w/2. < 0:
                Delta_x = int(image_width/focus_scale_w /2. -image_width/2.0)+1.


        # perception and decision
        decision_time = 1/LLM_FREQENCE if self.focus_idx <= 0 else (1/LLM_FREQENCE)/LLM_FOCUS_FACTOR
        # crop_img
        _, front_img_org = input_data.get('Center')
        _x = image_width / focus_scale_w /2.
        _y = image_height/ focus_scale_h /2.

        foucs_center_x = int(image_width / 2. + Delta_x)

        front_img = front_img_org[..., :-1][..., ::-1]
        front_img_pil = Image.fromarray(front_img)
        
        crop_img = front_img_pil.crop((int(foucs_center_x - _x), int(image_height/2. + Delta_y-_y), int(foucs_center_x + _x), int(image_height/2. + Delta_y+ _y)))
        
        nowTime = "Start"
        if timestamp - self.record_time >= decision_time:
            current_time = datetime.datetime.now()
            nowTime = f"{str(current_time.day).zfill(2)}{str(current_time.hour).zfill(2)}{str(current_time.minute).zfill(2)}{str(current_time.second).zfill(2)}"
            print('=========== Key frame {}===========\n'.format(nowTime))
            print(f'=> Current ID: {self.waypointer.current_idx}, Focus ID: {self.focus_idx}, LMM_DT: {decision_time}')
            # image sensor
            
            front_image_base64 = process_img(front_img)

            if self.focus_idx > 0:
                focus_view_base64 = process_img2(crop_img)
                # focus view
                focus_information = fetch_response_with_try_loop(image_base64=focus_view_base64)
                focus_box_with_ref, focus_box_with_ref_filter, focus_traffic_lights = fetch_all_box_with_ref(focus_information)
                self.focus_bbox = [x['box'] for x in focus_box_with_ref]
                focus_key_objects = process_information(focus_box_with_ref, focus_box_with_ref_filter, focus_traffic_lights, view='far-sighted view')
            else:
                focus_key_objects = 'No need to refer to the information in the far-sighted view.'

            # fetch all information
            # front view
            information = fetch_response_with_try_loop(image_base64=front_image_base64)
            box_with_ref, box_with_ref_filter, traffic_lights = fetch_all_box_with_ref(information)
            self.bbox = [x['box'] for x in box_with_ref]


            # stop sign
            notice_info = {}
            for x in box_with_ref:
                if 'stop sign' in x['ref']:
                    ret = re.search(r'(\d+)m', x['ref'])
                    if ret is not None:
                        dist = int(ret.group(1))
                        if dist < 5 and not self.stop_sign_flag:
                            notice_info['stop sign'] = True
                            if self.desired_speed == 0:
                                self.stop_sign_flag = True
                else:
                    self.stop_sign_flag = False
                            

            # scene description
            scene_description = {}
            key_objects = process_information(box_with_ref, box_with_ref_filter, traffic_lights, view='front view', focus=self.focus_idx>0) # filter traffic lights when not focusing
            scene_description['front_view'] = key_objects
            scene_description['focus_view'] = focus_key_objects # traffic lights
            
            # ego state
            ego_state = {}
            ego_state['speed'] = spd
            ego_state['steer'] = self.steer

            # notice_info
            if self.waypointer.global_plan[self.waypointer.current_idx-1][2] in [RoadOption.LEFT, RoadOption.RIGHT]:
                notice_info['turn'] = True

            ## don't stop for a long time
            if self.desired_speed==0:
                if self.stop_time == None:
                    self.stop_time = self.num_frame
                elif self.num_frame - self.stop_time > STOP_TIME:
                    notice_info['stop'] = True
            else:
                self.stop_time = None

            ## cross
            if self.focus_idx > 0:
                notice_info['cross'] = True
                
            message = aggregate_message(scene_description, ego_state, notice_info)
            print('=> Query Message: \n', ''.join([x['text'] for x in message[1:]]) if not LIGHT_LLM else ''.join([x['content']+'\n' for x in message]))

            gpt_on = GPT_ON and (len(box_with_ref_filter) or focus_key_objects != 'No need to refer to the information in the far-sighted view.') and not (self.desired_speed==0 and 'red' in focus_key_objects) if FAST_SIM else GPT_ON
            
            if LIGHT_LLM and FEW_SHOT > 0:
                query_embedding = fetch_embeddings_with_try_loop(fetch_scene_summary(message[-1]['content']))
                similar_scenes = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=FEW_SHOT
                )

                example_list = []
                for conv in similar_scenes['metadatas'][0]:
                    example_list += json.loads(conv['messages'])
                example_list += message
                message = example_list

            ans = fetch_response_with_try_loop(message=message) if gpt_on else 'Fake ## Decision\nIDLE'
            if not gpt_on and self.desired_speed < HIGH_SPEED and not (self.desired_speed==0 and 'red' in focus_key_objects):
                ans = 'Fake ## Decision\nAC'
            print('=> GPT Response: \n', ans)
            self.desired_speed = update_desired_speed(ans, self.desired_speed)
            print('=> Desired Speed: ', self.desired_speed)
            
            # update mistake queue
            if LIGHT_LLM:
                self.queue_for_reflection.append({"time": nowTime, "messages":[message[-1], {"role": "assistant", 'content': ans}], "embedding": query_embedding if FEW_SHOT>0 else None})
                if len(self.queue_for_reflection) > REFLECTION_SAMPLES:
                    self.queue_for_reflection.popleft() 
                        
            self.record_file.write('=========== Key frame {}===========\n'.format(nowTime))
            if LIGHT_LLM and FEW_SHOT > 0:
                self.record_file.write('Fewshot ids = {}\n'.format(similar_scenes['ids']))

            self.record_file.write('GPT query: {}\n'.format(''.join([x['text'] for x in message[1:]]) if not LIGHT_LLM else ''.join([x['role']+': '+x['content']+'\n' for x in message])))
            self.record_file.write('GPT ans: {}\n\n'.format(ans))
            self.num_frame += 1
            self.save_flag = True
            self.record_time = timestamp
            
            if self.pre_speed < 1e-5 and ego_state['speed'] < 1e-5:
                self.stop_flag = True
            else:
                self.stop_flag = False
            self.pre_speed = ego_state['speed']

        # Controler pure pursuit
        aim_idx = min(self.waypointer.current_idx+2,
                      len(self.waypointer.global_plan)-1)
        aimpoint_x, aimpoint_y = self.waypointer.global_plan[
            aim_idx][0], self.waypointer.global_plan[aim_idx][1]

        local_x, local_y = self.rotate_point(cur_x, cur_y, imu[6])
        local_aim_x, local_aim_y = self.rotate_point(
            aimpoint_x, aimpoint_y, imu[6])

        aim = (local_aim_x-local_x, local_aim_y-local_y)

        steer, throt, brake = self.pid_control(aim, spd, self.desired_speed)
        pid_control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
        self.steer = steer

        self._hic.run_interface(input_data, spd, pid_control, self.focus_bbox, self.bbox, self.num_frame, self.save_flag, self.unique_id, Delta_x, Delta_y, reflection_flag, accident_flag, crop_img, nowTime, self.stop_flag)
        self.save_flag = False
        self.odom_info['last_xy'] = (cur_x, cur_y)
        
        return pid_control

    def destroy(self):
        """
        Cleanup
        """
        self._hic._quit = True

        self.waypointer = None
        self.turn_controller = None
        self.speed_controller = None

