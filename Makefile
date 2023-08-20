PROJECT ?= md4all
VERSION ?= latest
DOCKER_IMAGE ?= $(PROJECT)/ubuntu_20.04-python_3.8-cuda_11.3.1-pytorch_1.13.1-gpu:${VERSION}

NOW=$(shell date '+%d.%m.%Y/%H:%M:%S')

ifeq ($(CPU1),)
CPU1 := $(CPU)
endif

SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--name ${PROJECT}-${NAME} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-e DATA_DIR=/mnt/data \
			-e CODE_DIR=/mnt/code \
			-u=<USER_ID>:<GROUP_ID> \
			-v ${HOME}:${HOME} \
			-v /etc/passwd:/etc/passwd:ro \
			-v /etc/group:/etc/group:ro \
			-v ~/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v <PATH_TO_DATAROOT>:/mnt/data \
			-v <PATH_TO_MD4ALL>:/mnt/code/md4all \
			--runtime=nvidia \
			--gpus all \
			--memory=55G \
			-w /mnt/code/md4all \
			--privileged \
			--ipc=host \
			--network=host \


NGPUS=$(shell nvidia-smi -L | wc -l)
MPI_CMD=mpirun \
		-allow-run-as-root \
		-np ${NGPUS} \
		-H localhost:${NGPUS} \
		-x MASTER_ADDR=127.0.0.1 \
		-x MASTER_PORT=23457 \
		-x HOROVOD_TIMELINE \
		-x OMP_NUM_THREADS=1 \
		-x KMP_AFFINITY='granularity=fine,compact,1,0' \
		-bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 \
		--report-bindings

####################### General #######################

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	docker build \
		-f Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-build-no-cache:
	docker build \
		-f Dockerfile \
		--no-cache \
		-t ${DOCKER_IMAGE} .
		
docker-start-interactive:
	docker run ${DOCKER_OPTS} --entrypoint bash ${DOCKER_IMAGE}

####################### Precomputation #######################

# RobotCar
docker-precompute-rgb-images-robotcar:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python data/robotcar/precompute_rgb_images.py --dataroot /mnt/data/robotcar --scenes 2014-12-09-13-21-02 2014-12-16-18-44-24 --camera_sensor stereo/left --out_dir /mnt/data/robotcar"

docker-precompute-pointcloud-robotcar:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python data/robotcar/precompute_depth_gt.py --dataroot /mnt/data/robotcar --scenes 2014-12-09-13-21-02 2014-12-16-18-44-24 --mode val test"

####################### Training #######################

# nuScenes
docker-train-baseline-nuscenes:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python train.py --config /mnt/code/md4all/config/train_baseline_nuscenes.yaml"

docker-train-md4allDDa-nuscenes:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python train.py --config /mnt/code/md4all/config/train_md4allDDa_nuscenes.yaml"

# RobotCar
docker-train-baseline-robotcar:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python train.py --config /mnt/code/md4all/config/train_baseline_robotcar.yaml"

docker-train-md4allDDa-robotcar:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python train.py --config /mnt/code/md4all/config/train_md4allDDa_robotcar.yaml"

####################### Evaluation #######################

# nuScenes
docker-eval-md4allDDa-80m-nuscenes-val:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python evaluation/evaluate_depth.py --config /mnt/code/md4all/config/eval_md4allDDa_80m_nuscenes_val.yaml"

docker-eval-md4allDDa-wo-daytime-norm-80m-nuscenes-val:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python evaluation/evaluate_depth.py --config /mnt/code/md4all/config/eval_md4allDDa_wo_daytime_norm_80m_nuscenes_val.yaml"

docker-eval-md4allDDa-wo-daytime-norm-80m-nuscenes-test:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python evaluation/evaluate_depth.py --config /mnt/code/md4all/config/eval_md4allDDa_wo_daytime_norm_80m_nuscenes_test.yaml"

# RobotCar
docker-eval-md4allDDa-50m-robotcar-test:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python evaluation/evaluate_depth.py --config /mnt/code/md4all/config/eval_md4allDDa_50m_robotcar_test.yaml"

docker-eval-md4allDDa-wo-daytime-norm-50m-robotcar-test:
	NOW=${NOW} docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
	/bin/bash -c "export PYTHONPATH="${PYTHONPATH}:/mnt/code/md4all" && python evaluation/evaluate_depth.py --config /mnt/code/md4all/config/eval_md4allDDa_wo_daytime_norm_50m_robotcar_test.yaml"
