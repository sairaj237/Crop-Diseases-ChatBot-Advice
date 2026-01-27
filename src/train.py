import mlflow
import mlflow.tensorflow

mlflow.start_run(run_name="baseline")

mlflow.log_param("img_size", 224)
mlflow.log_param("epochs", 20)
mlflow.log_metric("val_accuracy", val_acc)

mlflow.tensorflow.log_model(model, "model")

mlflow.end_run()
