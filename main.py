import mlflow.pytorch
import torch
import timm

model = timm.create_model("efficientnet_b0", num_classes=5)
model.load_state_dict(torch.load("models/baseline/efficientnet_sugarcane.pth"))
model.eval()

with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
    mlflow.register_model(
        "runs:/{}/model".format(mlflow.active_run().info.run_id),
        "sugarcane_disease_classifier"
    )
