"""Run backtest on trained model"""
import typer
import yaml
from pathlib import Path

from src.data.storage import DataStorage
from src.data.features import FeatureEngineer
from src.data.dataset import MLDatasetBuilder
from src.models.ensemble import EnsembleModel
from src.backtest.engine import BacktestEngine

app = typer.Typer()

@app.command()
def main(
    config_path: str = typer.Option('config/backtest_config.yaml'),
    model_path: str = typer.Option('models/ensemble_latest'),
    data_path: str = typer.Option('data/processed'),
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = EnsembleModel.load(Path(model_path))
    
    storage = DataStorage(Path(data_path))
    feature_eng = FeatureEngineer()
    
    dataset_builder = MLDatasetBuilder(
        storage=storage,
        feature_engineer=feature_eng,
        instruments=['EUR_USD', 'GBP_USD', 'USD_JPY'],
        granularity='H1',
    )
    
    dataset = dataset_builder.build_dataset()
    
    engine = BacktestEngine(config['backtest'])
    
    def strategy(bars, portfolio, prices):
        signals = []
        for inst in ['EUR_USD', 'GBP_USD', 'USD_JPY']:
            if inst in bars:
                latest = bars[inst]
                features = feature_eng.create_features(pd.DataFrame([latest])).iloc[[-1]]
                pred, _ = model.predict_with_proba(features)
                if pred[0] in [3, 4]:  # Buy signals
                    signals.append({'instrument': inst, 'side': 'buy', 'units': 0.1, 'tag': f'pred_{pred[0]}'})
                elif pred[0] in [0, 1]:  # Sell signals
                    signals.append({'instrument': inst, 'side': 'sell', 'units': 0.1, 'tag': f'pred_{pred[0]}'})
        return signals
    
    engine.set_strategy(strategy)
    engine.set_data({inst: storage.read(inst, 'H1') for inst in ['EUR_USD', 'GBP_USD', 'USD_JPY']})
    
    results = engine.run()
    
    typer.echo(f'Backtest Results:')
    for k, v in results.items():
        if k not in ['equity_curve', 'drawdown_series', 'trades']:
            typer.echo(f'  {k}: {v}')

if __name__ == '__main__':
    app()
