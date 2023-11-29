#!/usr/bin/env sh

cd $(dirname $(readlink -f $0))

docker image build --no-cache -t senti-core .

# TODO: Clean up any untagged docker images.

docker container run -it --rm \
    --name senti-core-devcontainer \
    --mount type=bind,src="$(pwd)",dst="/senti" \
    senti-core bash

    # TODO: Run training script in docker container