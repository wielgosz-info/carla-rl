# Carla RL Project

Hi, if you're looking for ready-to-use, documented, tested & stable OpenAI Gym that will work with CARLA, you should probably take a look at [macad-gym](https://github.com/praveen-palanisamy/macad-gym) or [OATomobile](https://github.com/oatml/oatomobile). If you're not particular about Gym, then try taking a look at [ScenarioRunner for CARLA
](https://github.com/carla-simulator/scenario_runner), which is prepared for the official [CARLA Challenge](https://leaderboard.carla.org/).

The purpose of this repo is to have an easy-to-run-on-headless-server setup in which we can arbitrarily modify anything that is needed for the current research angle. [Carla RL Project](https://github.com/carla-rl-gym/carla-rl) (which, as far as we were able to determine was created by people form [Machine Learning Group at University of Toronto](http://learning.cs.toronto.edu/) / [Waabi](https://waabi.ai/)) seemed to provide a good starting point for that, though we expect there may not be a lot of the original code left soon ;).

The intention is to be compatible with CARLA 0.9.x (more specifically, we use CARLA 0.9.11). Since the original benchmarks / reward functions etc. were designed for CARLA 0.8.x, they use measurements that are no longer available. We are doing some work to update them, but they **cannot** be used as a reference point or to "verify" how this or that benchmark actually works. We only wanted to get them up-and-running (i.e. have some agent that actually learns how to drive) and it is possible that they will get removed altogether one of those days.

Please note that at the moment only one CARLA server is supported. One can probably add more by editing `docker-compose.yml`, but this hasn't been tested yet.

Also, we have thrown-in some enhancements to simplify working with VS Code (e.g. volumes that store the remote VS Code or some additional Python packages). It should be pretty obvious which parts are that.

----------------
**TL;DR This code will change often and there is no documentation whatsoever aside from this README and whatever is in code. You have been warned.**

----------------

## Installation and Setup

### Running the CARLA Server, carlaviz & client (agents)
Docker compose file has been prepared for the ease of running the whole setup. It will honour several env variables, most important of which are probably `USER_ID` and `GROUP_ID` that ensure proper permissions for files in the mounted `client` dir. For details please see `.env` and `docker-compose.yml`.

### Arguments and Config Files
The `client/train.py` (or `/app/train.py` inside the container) script uses both arguments and a configuration file. The configuration file specifies all components of the model. The config file should have everything necessary to reproduce the results of a given model. The arguments of the script deal with things that are independent of the model (this includes things, like for example, ~~how often to create videos or~~ log to Tensorboard)

### Hyperparameter Tuning
~~To test a set of hyperparemeters see the `scripts/test_hyperparameters_parallel.py` script. This will let you specify a set of hyperparameters to test different from those specified in the `client/config/base.yaml` file.~~

The original script won't work at the moment. We will probably create working one someday.

## Benchmark Results

### A2C
~~To reproduce our results,~~ run a CARLA server and inside the `carla-rl_client_1` container run,
`python3 /app/train.py --config /app/config/a2c.yaml`

### ACKTR
~~To reproduce our results,~~ run a CARLA server and inside the `carla-rl_client_1` docker run,
`python /app/train.py --config /app/config/acktr.yaml`

### PPO
~~To reproduce our results,~~ run a CARLA server and inside the `carla-rl_client_1` docker run,
`python /app/train.py --config /app/config/ppo.yaml`

### On-Policy HER
~~To reproduce our results,~~ run a CARLA server and inside the `carla-rl_client_1` docker run,
`python /app/train.py --config /app/config/her.yaml`

## carlaviz

For in-browser visualisation to run correctly, the `carla-rl_viz_1` container's port `8081` needs to be accessible from YOUR web browser. This could probably be done by utilizing `CARLAVIZ_BACKEND_HOST` and `CARLAVIZ_BACKEND_PORT`, but since in our case the remote server is only accessible from the access node we need SSH port forwarding somewhere anyway. Used solution is: `carlaviz:8081` -> `remote_server:${CARLAVIZ_BACKEND_MAPPED_PORT:-49165}` -> `localhost:8081`

Example `.ssh/config` to achieve this:

```ssh-config
Host forward_carlaviz_backend
    HostName remote_server
    LocalForward 8081 127.0.0.1:49165
    User username
    ProxyJump username@access_node
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

and run (optionaly add `-v` to see what's happening):

```sh
ssh -N forward_carlaviz_backend
```

We could handle the `carlaviz:8080` port forwarding similarly, but VS Code can do it for us with a click of a button, so there's no need.