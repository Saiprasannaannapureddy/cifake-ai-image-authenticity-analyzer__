# Project Notes

## Changes in this version

1. Inference logic moved to `predict.py` so preprocessing and prediction are reusable.
2. Upload validation accepts only common image formats and limits files to 5 MB.
3. Uploaded files receive generated UUID names to reduce overwrite/path risks.
4. The trained model is loaded once and reused by the Flask process.
5. Training uses early stopping, learning-rate reduction, and best-checkpoint saving.
6. The web interface was redesigned with image preview and clearer result presentation.
7. The large local dataset is excluded from GitHub; setup instructions are documented instead.

## Important

The supplied trained model is retained so the application can run without retraining immediately. If you retrain after downloading the dataset, the model file will be replaced by the best checkpoint produced by `train_model.py`.
