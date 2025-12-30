 # Windows shell script
 conda activate GelSight-homemade
 python train.py --images marker_dataset\images\train --annotations marker_dataset\labels\train --format yolo --val_images marker_dataset\images\val --val_annotations marker_dataset\labels\val --epochs 100 --batch_size 6 --project centernet --lr 1e-5 