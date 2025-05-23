docker rm -f foundationpose
DIR=$(pwd)/../
echo "Starting container with DIR: $DIR"
xhost + && docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it --network=host --name foundationpose \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$DIR":"$DIR" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /data3:/data3 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY="${DISPLAY}" \
  -e GIT_INDEX_FILE \
  --workdir="$DIR" \
  foundationpose:latest bash
