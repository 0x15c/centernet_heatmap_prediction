# 3) Plot validation prediction vs GT + MAE
# Dataset - Training cross validation
python mlp_force_prediction/plot_mlp_validation.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_223951_MLP.pt `
  --checkpoint mlp_force_prediction/runs/session_20260311/mlp_force_model.pt `
  --output mlp_force_prediction/runs/cross_validation/A_A.png `
  --device auto

python mlp_force_prediction/plot_mlp_validation.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_223951_MLP.pt `
  --checkpoint mlp_force_prediction/runs/session_20260311_2/mlp_force_model.pt `
  --output mlp_force_prediction/runs/cross_validation/A_B.png `
  --device auto

python mlp_force_prediction/plot_mlp_validation.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_231504_MLP.pt `
  --checkpoint mlp_force_prediction/runs/session_20260311/mlp_force_model.pt `
  --output mlp_force_prediction/runs/cross_validation/B_A.png `
  --device auto

python mlp_force_prediction/plot_mlp_validation.py `
  --dataset mlp_force_prediction/datasets/Session_20260311_231504_MLP.pt `
  --checkpoint mlp_force_prediction/runs/session_20260311_2/mlp_force_model.pt `
  --output mlp_force_prediction/runs/cross_validation/B_B.png `
  --device auto