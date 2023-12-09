python dolos/methods/xception_attention/train_weak_supervision.py setup-a
python dolos/methods/xception_attention/predict.py -s weak -t setup-a -p celebahq-test
python dolos/methods/xception_attention/predict.py -s weak -t setup-a -p repaint-p2-test
python dolos/methods/xception_attention/evaluate.py -s weak -t setup-a -d repaint-p2