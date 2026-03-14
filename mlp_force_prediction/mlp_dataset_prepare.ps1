# 1) Generate dataset (Session_20260302_155834, status=Holding_Shear)
python mlp_force_prediction/generate_mlp_dataset.py `
  --force-jsonl mlp_force_prediction/calibration_data/Session_20260311_223951.jsonl `
  --displacement-jsonl mlp_force_prediction/data_from_frontend/Session_20260311_223951_MLP.jsonl `
  --output mlp_force_prediction/datasets/Session_20260311_223951_MLP.pt `
  --status "Holding_Shear" `
  --max-points 50 `
  --batch-size 64 `
  --val-ratio 0.25 `
  --seed 42 `
  --force-frame-offset 0 `
  --randomize-slots

python mlp_force_prediction/generate_mlp_dataset.py `
  --force-jsonl mlp_force_prediction/calibration_data/Session_20260311_231504.jsonl `
  --displacement-jsonl mlp_force_prediction/data_from_frontend/Session_20260311_231504_MLP.jsonl `
  --output mlp_force_prediction/datasets/Session_20260311_231504_MLP.pt `
  --status "Holding_Shear" `
  --max-points 50 `
  --batch-size 64 `
  --val-ratio 0.25 `
  --seed 42 `
  --force-frame-offset 0 `
  --randomize-slots