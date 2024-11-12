#!/bin/bash
set -e

echo "Pulling AlphaBind regressor model from S3"
aws s3 cp $MODEL_PATH model.pt

echo "Starting optimization"
python -m alphabind.optimizers.optimize_seeds --seed_sequence=$SEED_SEQUENCE --target=$TARGET_SEQUENCE --mutation_start_idx=$MUTATION_START_IDX --mutation_end_idx=$MUTATION_END_IDX --num_seeds=$NUM_SEEDS --num_generations=$NUM_GENERATIONS --trained_model_path=model.pt --output_file_path=optimized_seqs.csv --seed=$SEED --save_intermediate_steps=optimization_steps --batch_size=512 --generator_type=$GENERATOR

echo "Copying artifacts to S3"
aws s3 cp optimized_seqs.csv $OUTPUT_PATH/$GENERATOR/seed_$SEED/
aws s3 cp acceptance_rates.csv $OUTPUT_PATH/$GENERATOR/seed_$SEED/
aws s3 cp --recursive optimization_steps $OUTPUT_PATH/$GENERATOR/seed_$SEED/optimization_steps/
aws s3 cp ./alphabind/dist/alphabind-*-py3-none-any.whl $OUTPUT_PATH/$GENERATOR/seed_$SEED/

python -m alphabind.optimizers.merge_all_generations --intermediate_steps_path=optimization_steps/ --num_generations=$NUM_GENERATIONS
aws s3 cp all_unique_candidates.csv $OUTPUT_PATH/$GENERATOR/seed_$SEED/

echo "ALL FILES UPLOADED SUCCESSFULLY!"
