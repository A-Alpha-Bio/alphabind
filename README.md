# AlphaBind

**Pre-print available on bioRxiv: [AlphaBind, a Domain-Specific Model to Predict and Optimize Antibody-Antigen Binding Affinity](https://www.biorxiv.org/content/10.1101/2024.11.11.622872v1)**

<details open><summary><b>Table of contents</b></summary>

- [AlphaBind](#alphabind)
  - [Building the AlphaBind Docker Image](#building-the-alphabind-docker-image)
    - [Prerequisites](#prerequisites)
      - [Install Docker](#install-docker)
      - [Obtain a free NVIDIA NGC API key](#obtain-a-free-nvidia-ngc-api-key)
      - [Configure NGC credentials in the provided `.env` file](#configure-ngc-credentials-in-the-provided-env-file)
    - [Build the AlphaBind Docker Image](#build-the-alphabind-docker-image)
  - [Tutorials](#tutorials)
    - [Tutorial 1: Finetuning the AlphaBind Pre-trained Model Checkpoint and Running Inference](#tutorial-1-finetuning-the-alphabind-pre-trained-model-checkpoint-and-running-inference)
    - [Tutorial 2: Optimization Using a Fine-tuned Model](#tutorial-2-optimization-using-a-fine-tuned-model)
  - [Manuscript](#manuscript)
  - [License](#license)
    - [Copyright Disclaimer](#copyright-disclaimer)

</details>

## Building the AlphaBind Docker Image

### Prerequisites

#### Install Docker

To build the image, you will need a version of Docker with support for [buildkit Dockerfile syntax >= v1.10.0](https://docs.docker.com/build/buildkit/dockerfile-release-notes/#1100). Installation of Docker on various platforms is beyond the scope of this documentation, but instructions can be found in the [official Docker documentation](https://docs.docker.com/).

#### Obtain a free NVIDIA NGC API key

In order to download the model weights for the ESM-2nv model from NGC, you will need a free NGC account and `API Key`. If you do not already have these, create them by following NVIDIA's [NGC Account and API Key Configuration](https://docs.nvidia.com/bionemo-framework/1.10/access-startup.html#ngc-account-and-api-key-configuration) documentation.

#### Configure NGC credentials in the provided `.env` file

Securely set the `NGC_CLI_API_KEY` and `NGC_CLI_ORG` environment variables on the host system using the credentials obtained above.

If performing this step manually, security best practice is to avoid persisting these values in your shell history. One approach to doing so is to populate the provided [ngc_secrets.env.template](.ngc_secrets.env.template) file with your API credentials, rename it to `ngc_secrets.env`, then export the environment variables in that file by running:
```[host shell]
set -a
source ./ngc_secrets.env
set +a
```

### Build the AlphaBind Docker Image

> [!IMPORTANT]
> If you previously set the environment variables in the preceding section using the `source ./ngc_secrets.env` method, the following command must be run in that same shell session (or a subshell thereof).

```[host shell]
# Extract the alphabind Python package version from our pyproject.toml
ALPHABIND_VERSION=$(sed -n 's/.*version = "\([^"]*\)".*/\1/p' ./alphabind/pyproject.toml)

docker build --secret id=NGC_CLI_API_KEY --secret id=NGC_CLI_ORG -t alphabind:latest -t alphabind:${ALPHABIND_VERSION} .
```

## Tutorials

We recommend that most users start with our two tutorial notebooks to familiarize themselves with terminology and usage details.

### [Tutorial 1: Finetuning the AlphaBind Pre-trained Model Checkpoint and Running Inference](alphabind/examples/finetuning_and_inference/tutorial_1_finetuning_alphabind.ipynb)

This tutorial details a usage example for fine-tuning the AlphaBind pre-trained checkpoint using a third-party dataset. Additional details are available in the tutorial's accompanying [README](alphabind/examples/finetuning_and_inference/README.md).

### [Tutorial 2: Optimization Using a Fine-tuned Model](alphabind/examples/optimization/tutorial_2_optimization_alphabind.ipynb)

> [!IMPORTANT]
> This tutorial depends on prior successful completion of Tutorial 1.

This tutorial details a usage example for optimizing a parental sequence against a target using the fine-tuned model trained in Tutorial 1. Additional details are available in the tutorial's accompanying [README](alphabind/examples/README.md).

## Manuscript

**Pre-print available on bioRxiv: [AlphaBind, a Domain-Specific Model to Predict and Optimize Antibody-Antigen Binding Affinity](https://www.biorxiv.org/content/10.1101/2024.11.11.622872v1)**

Figure plotting code, data, and representative production commands associated with our manuscript can be found in the [`manuscript`](./manuscript) directory. Note that a few legend naming conventions were cosmetically aliased for our pre-print, downstream of the [plotting code](manuscript/notebooks/manuscript_analysis.ipynb) in this repository, but figure content is identical.

## License

See [LICENSE](./LICENSE).

### Copyright Disclaimer

All product names, logos, and brands mentioned in this documentation are property of their respective owners. "NVIDIA", "NGC", and "BioNeMo" are trademarks or registered trademarks of NVIDIA Corporation. "Docker" is a trademark or registered trademark of Docker, Inc. The use of these names, logos, and brands does not imply endorsement.

© A-Alpha Bio, Inc. 2024. All rights reserved.
