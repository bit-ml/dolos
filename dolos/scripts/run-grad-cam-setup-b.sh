python dolos/methods/grad_cam/train_weak_supervision.py setup-b
python dolos/methods/grad_cam/predict.py -s weak -t setup-b -p celebahq-test
python dolos/methods/grad_cam/predict.py -s weak -t setup-b -p repaint-p2-test
python dolos/methods/grad_cam/evaluate.py -s weak -t setup-b -d repaint-p2