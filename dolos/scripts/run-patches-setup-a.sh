python dolos/methods/patch_forensics/train_weak_supervision.py setup-a
python dolos/methods/patch_forensics/predict.py -s weak -t setup-a -p celebahq-test
python dolos/methods/patch_forensics/predict.py -s weak -t setup-a -p repaint-p2-test
python dolos/methods/patch_forensics/evaluate.py -s weak -t setup-a -d repaint-p2