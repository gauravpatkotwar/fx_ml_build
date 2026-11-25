"""Train ML models with MLflow tracking"""
import typer
import mlflow
import yaml
from pathlib import Path
from datetime import datetime

from src.data.storage import DataStorage
from src.data.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry

app = typer.Typer()

@app.command()
def main(
    config_path: str = typer.Option('config/model_config.yaml'),
    data_path: str = typer.Option('data/processed'),
    model_output: str = typer.Option('models'),
    experiment: str = typer.Option('fx-ml-training'),
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    mlflow.set_experiment(experiment)
    
    storage = DataStorage(Path(data_path))
    feature_eng = FeatureEngineer()
    
    trainer = ModelTrainer(config['training'], config['models'])
    
    with mlflow.start_run(run_name=f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}') as run:
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        mlflow.log_params(flatten_dict(config))
        
        model, metrics = trainer.train_ensemble(storage, feature_eng)
        
        mlflow.log_metrics(metrics)
        
        registry = ModelRegistry(Path(model_output))
        version = registry.register_model(model, metrics, run.info.run_id)
        
        typer.echo(f'Model registered as version {version}')
        typer.echo(f'Metrics: {metrics}')

if __name__ == '__main__':
    app()
