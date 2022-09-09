#!/bin/bash

trap 'killall' INT TERM

killall() {
    trap '' INT TERM     # ignore INT and TERM while shutting down
    echo "**** Shutting down... ****"     # added double quotes
    # kill -TERM 0
    kill -TERM -$(ps -o pgid=$$ | grep -o '[0-9]*')
    wait
    echo "DONE"
    im-remind qq "Task($PROJECT_NAME-$PROJECT_VERSION) is killed."
}

get_unused_port() {
    # find a free port
    while
        local port=$(shuf -n 1 -i 50000-65535)
        netstat -atun | grep -q "$port"
    do
        continue
    done
    echo $port
}

cd "$(dirname $0)"

echo "Start Task($PROJECT_NAME-$PROJECT_VERSION)..."

_COMMA_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd "," | wc -c)
GPU_NUM=$(($_COMMA_NUM + 1))

# fix seeds
export PL_GLOBAL_SEED=42
export PL_SEED_WORKERS=1

export MASTER_ADDR=localhost
export MASTER_PORT=$(get_unused_port)
# number of processes
export WORLD_SIZE=$GPU_NUM

# slave nodes
for i in $(seq $(($GPU_NUM-1)) -1 1)
do
    NODE_RANK=0 LOCAL_RANK=$i python $SCRIPT_PATH \
    -c $CONFIG_PATH \
    -v $PROJECT_VERSION \
    -n $PROJECT_NAME \
    --gpus $GPU_NUM \
    --stage $STAGE \
    $@ &
done
# master node
NODE_RANK=0 LOCAL_RANK=0 python $SCRIPT_PATH \
    -c $CONFIG_PATH \
    -v $PROJECT_VERSION \
    -n $PROJECT_NAME \
    --gpus $GPU_NUM \
    --stage $STAGE \
    $@

if [ $? -eq "0" ]; then
    im-remind qq "Task($PROJECT_NAME-$PROJECT_VERSION) succeed."
else
    im-remind qq "Task($PROJECT_NAME-$PROJECT_VERSION) failed."
fi
