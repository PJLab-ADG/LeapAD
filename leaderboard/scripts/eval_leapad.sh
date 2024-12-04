#!/bin/bash
carla_port=$1
traffic_port=$2
vlm_port=$3
llm_port=$4
echo "carla port = ${carla_port}"
echo "traffic_port = ${traffic_port}"
echo "vlm port =  ${vlm_port}"
echo "llm port = ${llm_port}"
export CARLA_ROOT=/path.../CARLA_0.9.15
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg":"${SUMO_HOME}/tools/":${PYTHONPATH}
export LEADERBOARD_ROOT=leaderboard
export SCENARIO_RUNNER_ROOT=scenario_runner
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${SUMO_HOME}/tools/":${PYTHONPATH}
export TEAM_AGENT=team_code/leapad_agent.py
export CHECKPOINT_ENDPOINT=result/test.json
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export REPETITIONS=1
export PORT=${carla_port}
export TM_PORT=${traffic_port} # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export CHALLENGE_TRACK_CODENAME=SENSORS
export ROUTES=${LEADERBOARD_ROOT}/data/routes_testing.xml # change the xml file for different routes
export HUMAN_AGENT="LeanAD Agent"
export VLM_PORT=${vlm_port}
export LLM_PORT=${llm_port}


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--debug=${DEBUG_CHALLENGE} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

