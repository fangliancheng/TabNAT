#!/bin/bash

datasets=(adult beijing car default magic news nursery shoppers)

# Training
for dataname in "${datasets[@]}"
do
  echo "Training on dataset: $dataname"
  python main.py --dataname "$dataname" --method tabnat --mode train
done

# Sampling
for dataname in "${datasets[@]}"
do
  echo "Sampling on dataset: $dataname"
  python main.py --dataname "$dataname" --method tabnat --mode sample
done

# Evaluation
echo "Starting evaluation at $(date)" > evaluation_results.txt
for dataname in "${datasets[@]}"
do
  echo "Evaluating on dataset: $dataname"
  echo "=== Results for $dataname ===" >> evaluation_results.txt
  
  echo "Running density evaluation..." >> evaluation_results.txt
  python eval/eval_density.py --dataname "$dataname" --path synthetic/"$dataname"/tabdar.csv >> evaluation_results.txt
  echo "" >> evaluation_results.txt
  
  echo "Running quality evaluation..." >> evaluation_results.txt
  python eval/eval_quality.py --dataname "$dataname" --path synthetic/"$dataname"/tabdar.csv >> evaluation_results.txt
  echo "" >> evaluation_results.txt
  
  echo "Running detection evaluation..." >> evaluation_results.txt
  python eval/eval_detection.py --dataname "$dataname" --path synthetic/"$dataname"/tabdar.csv >> evaluation_results.txt
  echo "" >> evaluation_results.txt
  
  echo "Running JSD evaluation..." >> evaluation_results.txt
  python eval/eval_jsd.py --dataname "$dataname" --path synthetic/"$dataname"/tabdar.csv >> evaluation_results.txt
  echo "" >> evaluation_results.txt
  
  echo "Running MLE evaluation..." >> evaluation_results.txt
  python eval/eval_mle.py --dataname "$dataname" --path synthetic/"$dataname"/tabdar.csv >> evaluation_results.txt
  echo "" >> evaluation_results.txt
  
  echo "----------------------------------------" >> evaluation_results.txt
done
echo "Evaluation completed at $(date)" >> evaluation_results.txt