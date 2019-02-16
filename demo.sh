set -e

python -u main.py --method l12rmsg --nepochs 20 --log_interval 1
python -u main.py --method incremental --nepochs 20 --log_interval 1
python -u main.py --method sgd --nepochs 20 --log_interval 1
python -u main.py --method original  