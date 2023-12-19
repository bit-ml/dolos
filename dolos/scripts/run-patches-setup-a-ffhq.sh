python dolos/methods/patch_forensics/train_weak_supervision.py setup-a-ffhq
python dolos/methods/patch_forensics/predict.py -s weak -t setup-a-ffhq -p celebahq-test
python dolos/methods/patch_forensics/predict.py -s weak -t setup-a-ffhq -p repaint-p2-test
python dolos/methods/patch_forensics/evaluate.py -s weak -t setup-a-ffhq -d repaint-p2