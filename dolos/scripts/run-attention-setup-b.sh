python dolos/methods/xception_attention/train_weak_supervision.py setup-b
python dolos/methods/xception_attention/predict.py -s weak -t setup-b -p celebahq-test
python dolos/methods/xception_attention/predict.py -s weak -t setup-b -p repaint-p2-test
python dolos/methods/xception_attention/evaluate.py -s weak -t setup-b -d repaint-p2