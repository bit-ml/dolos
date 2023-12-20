python dolos/methods/grad_cam/train_weak_supervision.py setup-a
python dolos/methods/grad_cam/predict.py -s weak -t setup-a -p celebahq-test
python dolos/methods/grad_cam/predict.py -s weak -t setup-a -p repaint-p2-test
python dolos/methods/grad_cam/evaluate.py -s weak -t setup-a -d repaint-p2