set -e

python main.py --method l12rmsg --nepochs 20 --log_interval 1
python main.py --method incremental --nepochs 20 --log_interval 1
python main.py --method sgd --nepochs 20 --log_interval 1
python main.py --method original  