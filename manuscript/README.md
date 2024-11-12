# AlphaBind

- [AlphaBind](#alphabind)
  - [Selected production commands used to produce experimental data](#selected-production-commands-used-to-produce-experimental-data)
    - [Pre-compute ESM-2nv embeddings](#pre-compute-esm-2nv-embeddings)
    - [Fine-tune the applicable pre-trained model on target-specific data](#fine-tune-the-applicable-pre-trained-model-on-target-specific-data)
    - [Generate optimized candidates via MCMC](#generate-optimized-candidates-via-mcmc)
      - [ESM-Mask Strategy (`esm-simultaneous-random`)](#esm-mask-strategy-esm-simultaneous-random)
        - [TIGIT](#tigit)
        - [Pembro](#pembro)
        - [COVID (VHH72)](#covid-vhh72)
        - [Pembro852](#pembro852)
        - [Trastuzumab (Full)](#trastuzumab-full)
        - [Trastuzumab (CDR3-Only)](#trastuzumab-cdr3-only)
    - [Merge candidates from all generations into a single dataset](#merge-candidates-from-all-generations-into-a-single-dataset)


## Selected representative production commands used to produce experimental data

### Pre-compute ESM-2nv embeddings

```
python -m alphabind.features.build_features --input_filepath=mason_data.csv --output_filepath=mason_data_featurized.csv --embedding_dir_path=embeddings_dir
```

### Fine-tune the applicable pre-trained model on target-specific data

```
python -m alphabind.models.train_model --dataset_csv_path=mason_data_featurized.csv --tx_model_path=alphabind.pt --max_epochs=100 --output_model_path=finetuned_alphabind.pt
```

### Generate optimized candidates via MCMC

#### ESM-Mask Strategy (`esm-simultaneous-random`)

**NOTE:** These commands were run from `manuscript/scripts/optimize`

##### TIGIT
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/TIGIT/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/TIGIT/esm_warm 7 ../../conf/targets/tigit_variables.env esm-simultaneous-random 100 7500
```

##### Pembro
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/Pembro/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/Pembro/esm_warm 7 ../../conf/targets/pembro_variables.env esm-simultaneous-random 100 7500
```

##### COVID (VHH72)
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/Covid/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/Covid/esm_warm 7 ../../conf/targets/covid_variables.env esm-simultaneous-random 100 7500
```

##### Pembro852
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/Pembro852/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/Pembro852/esm_warm 7 ../../conf/targets/pembro_variables.env esm-simultaneous-random 100 7500
```

##### Trastuzumab (Full)
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/Trastuzumab/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/Trastuzumab/full/esm_warm 7 ../../conf/targets/trastuzumab_full_variables.env esm-simultaneous-random 100 7500
```

##### Trastuzumab (CDR3-Only)
```
./docker_start_p5s.sh s3://aalphabio-public-data/benchmarks/Run3/Trastuzumab/esm_warm/model.pt s3://aalphabio-public-data/benchmarks/ESM_Run2_simultaneous/Trastuzumab/cdr/esm_warm 7 ../../conf/targets/trastuzumab_cdr_variables.env esm-simultaneous-random 100 7500
```

### Merge candidates from all generations into a single dataset

```
python -m alphabind.optimizers.merge_all_generations --intermediate_steps_path optimization_steps --num_generations 100
```
