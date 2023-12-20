python dolos/methods/grad_cam/train_full_supervision.py setup-c
python dolos/methods/grad_cam/predict.py -s full -t setup-c -p repaint-p2-test
python dolos/methods/grad_cam/evaluate.py -s full -t setup-c -d repaint-p2