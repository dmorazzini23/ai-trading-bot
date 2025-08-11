import logging

from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig
CONFIG = TradingConfig()

# Map old names if needed
set_runtime_config = getattr(config, "set_runtime_config", None)
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/update_config", methods=["POST"])
def update_config():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        volume_thr = float(data.get("volume_spike_threshold", 1.5))
        ml_thr = float(data.get("ml_confidence_threshold", 0.5))

        if not (0.1 <= volume_thr <= 10.0):
            return jsonify({"error": "volume_spike_threshold must be between 0.1 and 10.0"}), 400
        if not (0.0 <= ml_thr <= 1.0):
            return jsonify({"error": "ml_confidence_threshold must be between 0.0 and 1.0"}), 400

        pyramid_levels = data.get(
            "pyramid_levels",
            {"high": 0.4, "medium": 0.25, "low": 0.15},
        )

        for level, value in pyramid_levels.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                return jsonify({"error": f"Invalid pyramid level {level}: {value}"}), 400

        set_runtime_config(volume_thr, ml_thr, pyramid_levels)
        return jsonify(
            {
                "status": "updated",
                "volume_spike_threshold": volume_thr,
                "ml_confidence_threshold": ml_thr,
                "pyramid_levels": pyramid_levels,
            }
        )

    except ValueError as e:
        return jsonify({"error": f"Invalid number format: {str(e)}"}), 400
    except Exception:
        logging.exception("Config update failed")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

