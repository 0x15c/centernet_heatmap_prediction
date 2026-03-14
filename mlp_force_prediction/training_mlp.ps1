# 2) Train MLP
python mlp_force_prediction/train_mlp_force_regressor.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_223951_MLP.pt `
  --output-dir mlp_force_prediction/runs/Session_20260311 `
  --epochs 1000 `
  --lr 1e-3 `
  --weight-decay 0.001 `
  --device auto `
  --batch-size 64
python mlp_force_prediction/train_mlp_force_regressor.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_231504_MLP.pt `
  --output-dir mlp_force_prediction/runs/Session_20260311_2 `
  --epochs 1000 `
  --lr 1e-3 `
  --weight-decay 0.001 `
  --device auto `
  --batch-size 64