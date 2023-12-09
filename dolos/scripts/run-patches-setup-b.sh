python dolos/methods/patch_forensics/train_weak_supervision.py setup-b
python dolos/methods/patch_forensics/predict.py -s weak -t setup-b -p celebahq-test
python dolos/methods/patch_forensics/predict.py -s weak -t setup-b -p repaint-p2-test
python dolos/methods/patch_forensics/evaluate.py -s weak -t setup-b -d repaint-p2