"""Model Context Protocol (MCP) Server for FX ML Model Serving"""
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult
from pydantic import BaseModel

from src.models.ensemble import EnsembleModel
from src.data.features import FeatureEngineer
from src.data.storage import DataStorage

logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self, model_path: str, data_path: str):
        self.server = Server("fx-ml-trading")
        self.model = EnsembleModel.load(model_path)
        self.feature_engineer = FeatureEngineer()
        self.data_path = Path(data_path)
        self.storage = DataStorage(Path(data_path))
        self._register_tools()

    def _register_tools(self):
        @self.server.tool()
        async def get_prediction(symbol: str, timeframe: str = "H1", bars: int = 500) -> CallToolResult:
            try:
                df = self.storage.read(symbol, timeframe)
                if df is None or len(df) < 100:
                    return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"Insufficient data for {symbol}"}))])

                df = self.feature_engineer.create_features(df)
                feature_cols = [c for c in df.columns if c not in ["timestamp", "instrument", "granularity"]]
                latest = df[feature_cols].iloc[[-1]].fillna(method="ffill").fillna(0)

                prediction, probabilities = self.model.predict_with_proba(latest)
                confidence = float(max(probabilities[0]))

                signal_map = {0: "STRONG_SELL", 1: "SELL", 2: "HOLD", 3: "BUY", 4: "STRONG_BUY"}

                result = {
                    "symbol": symbol, "timeframe": timeframe,
                    "timestamp": datetime.utcnow().isoformat(),
                    "prediction": int(prediction),
                    "signal": signal_map.get(int(prediction), "UNKNOWN"),
                    "confidence": confidence,
                    "probabilities": {
                        "STRONG_SELL": float(probabilities[0][0]),
                        "SELL": float(probabilities[0][1]),
                        "HOLD": float(probabilities[0][2]),
                        "BUY": float(probabilities[0][3]),
                        "STRONG_BUY": float(probabilities[0][4])
                    }
                }
                return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))])

        @self.server.tool()
        async def get_feature_importance(model_version: str, top_n: int = 20) -> CallToolResult:
            try:
                importance = self.model.get_feature_importance(top_n)
                result = {
                    "model_version": model_version,
                    "features": [{"feature": f, "importance": float(imp)} for f, imp in importance]
                }
                return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])
            except Exception as e:
                return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))])

        @self.server.tool()
        async def trigger_retrain(reason: str = "scheduled") -> CallToolResult:
            result = {
                "status": "triggered", "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "job_id": f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            }
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    def run(self, host: str = "0.0.0.0", port: int = 8001):
        logger.info(f"Starting MCP server on {host}:{port}")
        self.server.run(host=host, port=port)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ensemble_latest"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    server = MCPServer(model_path, data_path)
    server.run()