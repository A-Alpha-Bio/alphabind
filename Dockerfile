# syntax=docker.io/docker/dockerfile:1.10
FROM nvcr.io/nvidia/clara/bionemo-framework:1.10
WORKDIR /workspace/bionemo

# `NGC_CLI_TEAM` needs to be defined, but can typically be left empty for most use-cases.
ARG NGC_CLI_TEAM=
# Download the ESM-2nv, 3B parameter model.
RUN --mount=type=secret,id=NGC_CLI_API_KEY,env=NGC_CLI_API_KEY \
    --mount=type=secret,id=NGC_CLI_ORG,env=NGC_CLI_ORG \
    python download_artifacts.py --model_dir ${BIONEMO_HOME}/models --models esm2nv_3b

RUN pip install --no-cache-dir build

COPY ./alphabind ./alphabind

# NOTE: The constraints.txt file ensures that we preserve the exact versions of
# low-level ML dependencies distributed in the base image.
RUN set -o pipefail \
    && cd alphabind \
    && python -m build \
    && python -m pip list --format freeze | egrep "torch|nvidia|cuda" > constraints.txt \
    && python -m pip install --no-cache-dir -c constraints.txt ./dist/alphabind-*-py3-*.whl
