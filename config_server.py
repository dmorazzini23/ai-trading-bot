from flask import Flask, request, jsonify
from config import set_runtime_config

app = Flask(__name__)


@app.route("/update_config", methods=["POST"])
def update_config():
    data = request.get_json()
    volume_thr = float(data.get("volume_spike_threshold", 1.5))
    ml_thr = float(data.get("ml_confidence_threshold", 0.5))
    pyramid_levels = data.get(
        "pyramid_levels",
        {"high": 0.4, "medium": 0.25, "low": 0.15},
    )

    set_runtime_config(volume_thr, ml_thr, pyramid_levels)
    return jsonify(
        {
            "status": "updated",
            "volume_spike_threshold": volume_thr,
            "ml_confidence_threshold": ml_thr,
            "pyramid_levels": pyramid_levels,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

