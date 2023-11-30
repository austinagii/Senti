#!/usr/bin/env sh

cd "$(dirname "$(readlink -f "$0")")"

docker image build --no-cache -t senti-core .

if [[ $1 == "train" ]]; then 
    COMMAND="pipenv sync && pipenv run python ./scripts/train.py"
fi

if [[ -n $COMMAND ]]; then
    docker container run -it --rm \
        --name senti-core-devcontainer \
        --mount type=bind,src="$(pwd)",dst="/senti" \
        senti-core /usr/bin/env bash -c "$COMMAND"
else
    docker container run -it --rm \
        --name senti-core-devcontainer \
        --mount type=bind,src="$(pwd)",dst="/senti" \
        senti-core /usr/bin/env bash
fi

# TODO: Investigate reduced training performance on docker container
