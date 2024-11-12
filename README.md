# AlphaBind

## Building the AlphaBind Docker container

> [!TIP]
> For convenience, we provide a script to automate the below setup process (other than the [Prerequisites](#prerequisites)). It can be run from the repo root directory as:
>    ```
>    bash ./alphabind/scripts/build.sh
>    ```

### Prerequisites

#### Obtain a free NVIDIA NGC API key

In order to download the model weights for the ESM-2nv model from NGC, you will need a free NGC account and `API Key`. If you do not already have these, create them by following NVIDIA's [NGC Account and API Key Configuration](https://docs.nvidia.com/bionemo-framework/latest/access-startup.html#ngc-account-and-api-key-configuration) documentation.

#### Configure NGC credentials in the provided `.env` file

Securely set the `NGC_CLI_API_KEY` and `NGC_CLI_ORG` environment variables on the host system using the credentials obtained above.

If performing this step manually, security best practice is to avoid persisting these values in your shell history. One approach to doing so is to populate the provided [ngc_secrets.env.template](.ngc_secrets.env.template) file with your API credentials, rename it to `ngc_secrets.env`, then export the environment variables in that file by running:
```[host shell]
set -a
source ./ngc_secrets.env
set +a
```

### Build the AlphaBind Docker Container

**NOTE:** If you previously set the environment variables in the preceding section using the `source ./ngc_secrets.env` method, the following command must be run in that same shell session (or a subshell thereof).

```[host shell]
ALPHABIND_VERSION=$(grep -Po '(?<=version = ")[^"]*' ./alphabind/pyproject.toml)
docker build --secret id=NGC_CLI_API_KEY --secret id=NGC_CLI_ORG -t alphabind:latest -t alphabind:${ALPHABIND_VERSION} .
```

## License

See [LICENSE](./LICENSE).

### Copyright Disclaimer

All product names, logos, and brands mentioned in this documentation are property of their respective owners. "NVIDIA", "NGC", and "BioNeMo" are trademarks or registered trademarks of NVIDIA Corporation. "Docker" is a trademark or registered trademark of Docker, Inc. The use of these names, logos, and brands does not imply endorsement.

© A-Alpha Bio, Inc. 2024. All rights reserved.
