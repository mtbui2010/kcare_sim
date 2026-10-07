# Host-side targets (run from ~/trung_workdir/kaair_sim/sim) and in-container
# targets (run inside kaair_sim_humble, cwd /sim). VARIANT = slamtec | clober.
VARIANT ?= slamtec

# ── host ──────────────────────────────────────────────────────────────────────
image:        ## build kaair-moveit:humble (repo Dockerfile) then kaair-sim:humble
	cd ../keti_assist_ai_robot_ros2 && bash scripts/build_docker.sh -t humble
	docker build -t kaair-sim:humble -f docker/Dockerfile.sim docker
up:           ## start the container
	docker compose up -d
shell:        ## open a shell in it
	docker exec -it kaair_sim_humble bash

# ── container ─────────────────────────────────────────────────────────────────
ws:           ## import xarm/camera repos + build description packages (once)
	bash scripts/setup_ws.sh
urdf:         ## xacro -> urdf/raw/kaair_{slamtec,clober}.urdf
	bash scripts/flatten_urdf.sh
audit:        ## what the raw URDF is missing for physics
	python3 scripts/build_sim_urdf.py $(VARIANT) --audit
sim-urdf:     ## raw -> urdf/kaair_$(VARIANT)_sim.urdf (+ assets/)
	python3 scripts/build_sim_urdf.py $(VARIANT)
mjcf:         ## sim urdf -> mjcf/kaair_$(VARIANT).xml + tabletop scene
	python3 scripts/urdf_to_mjcf.py $(VARIANT)
check:        ## FK / hold / step / grasp / render -> out/$(VARIANT)/
	python3 scripts/check_sim.py $(VARIANT)
all: urdf sim-urdf mjcf check

.PHONY: image up shell ws urdf audit sim-urdf mjcf check all
