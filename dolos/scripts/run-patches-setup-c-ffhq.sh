python dolos/methods/patch_forensics/train_full_supervision.py setup-c-ffhq
python dolos/methods/patch_forensics/predict.py -s full -t setup-c-ffhq -p repaint-p2-test
python dolos/methods/patch_forensics/evaluate.py -s full -t setup-c-ffhq -d repaint-p2