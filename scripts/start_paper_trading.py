"""Start paper trading on OANDA demo account"""
import sys
sys.path.insert(0, '.')

from src.paper_trading.simulator import PaperTradingSimulator, PaperTradingConfig
import yaml

if __name__ == '__main__':
    with open('config/paper_trading_config.yaml') as f:
        config_dict = yaml.safe_load(f)
    config = PaperTradingConfig(**config_dict['paper_trading'])
    simulator = PaperTradingSimulator(config)
    simulator.run()
