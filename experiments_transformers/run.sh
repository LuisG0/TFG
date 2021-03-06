GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('../data/M3')
MODELS=('trD_AR')
PARAMETERS=./parameters.json
OUTPUT=./results
CSV_FILENAME=results.csv

python main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
