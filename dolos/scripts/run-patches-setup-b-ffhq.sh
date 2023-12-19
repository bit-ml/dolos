python dolos/methods/patch_forensics/train_weak_supervision.py setup-b-ffhq
python dolos/methods/patch_forensics/predict.py -s weak -t setup-b-ffhq -p celebahq-test
python dolos/methods/patch_forensics/predict.py -s weak -t setup-b-ffhq -p repaint-p2-test
python dolos/methods/patch_forensics/evaluate.py -s weak -t setup-b-ffhq -d repaint-p2