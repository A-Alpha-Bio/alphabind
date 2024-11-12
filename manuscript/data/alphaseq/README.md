# AlphaSeq Data Guide

AlphaSeq fine-tuning and validation datasets are provided in .csv format, with one row per protein-protein interaction (PPI) in the dataset, and with the following columns:

- `mata_description`: an identifier for the protein of interest (POI) displayed on the Mat-A yeast library for this PPI

- `matalpha_description`: an identifier for the POI displayed on the Mat-Alpha yeast library for this PPI

- `alphaseq_affinity`: the AlphaSeq-reported affinity for the PPI between mata_description and matalpha_description, reported as $log_{10}(\mathrm{K_D}\,\mathrm{(nM)})$, i.e. a value of 0 corresponds to 1 nM, a value of 1 corresponds to 10 nM, etc.

- `mata_sequence`: the amino acid sequence of the POI displayed on the Mat-A yeast library for this PPI. Note that to reduce file size, ‘matalpha_sequence’ is not reported here, but the Mat-Alpha POI sequence is invariant for all rows and is the target construct for each antibody campaign (see Table S1 for target amino acid sequences)

- `edit_distance`: the Levenshtein (edit) distance between ‘mata_sequence’ and its associated parental antibody

- `parental_mean_aff`: the mean AlphaSeq-reported affinity for all replicates of the associated parental antibody

- `parental_aff_margin`: the difference in AlphaSeq-reported affinity between this PPI and its associated parental antibody i.e. ‘alphaseq_affinity’ – ‘parental_mean_aff’

- `predicted_affinity` (only for validation datasets): the predicted affinity for this PPI, according to ‘model’

- `model` (only for validation datasets): the model that was used to generate ‘mata_description’ as a candidate, and supplied the ‘predicted_affinity’.  ‘model’ will be one of: ohe_cold (one hot encoded, no pretraining); ‘ohe_warm’ (one hot encoded, pretraining); ‘esm_cold’ (ESM-2nv embeddings, no pretraining); ‘esm_warm’ (ESM-2nv embeddings, pretraining); ‘alphabind’ (the full AlphaBind model, same regressor as ‘esm_warm’ with ESM-2nv masking for sequence perturbation during optimization)

- `id` (only for validation datasets): a numerical identifier for the different candidates from each model, unique within each AlphaSeq dataset
