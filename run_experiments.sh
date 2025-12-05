cfile=$1
name=$2
shift 2
for i in {1..8}
do
python -m src.reconstruct.main -c $cfile --experiment_name temp --train_epochs 3 "$@"
mv logs/temp logs/run_${name}_$i
done
