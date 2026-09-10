
import json
import math
import statistics
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import pandas as pd

MODEL_FILE = "sensor_tool_condition_model.pkl"
HOST = "0.0.0.0"
PORT = 8765

MODEL_FEATURES = [
    "F_f_RMS","F_f_MEAN","F_f_MAX",
    "F_c_RMS","F_c_MEAN","F_c_MAX",
    "F_p_RMS","F_p_MEAN","F_p_MAX",
    "CV3.Z_MEAN","CV3.Z_RMS",
    "TV2.Z_MEAN","TV2.Z_RMS",
    "AE_RMS","AE_C_0.1",
    "CON.G.FREAL","CON.A.SREAL.S",
]

model = joblib.load(MODEL_FILE)

def stats(values):
    if not values:
        return 0.0, 0.0, 0.0
    vals = [float(x) for x in values]
    mean = sum(vals) / len(vals)
    rms = math.sqrt(sum(x*x for x in vals) / len(vals))
    peak = max(abs(x) for x in vals)
    return mean, rms, peak

def condition(wear):
    if wear < 0.15:
        return "HEALTHY"
    if wear < 0.20:
        return "MONITOR"
    if wear < 0.25:
        return "WARNING"
    return "TOOL CHANGE"

def from_features(d):
    # Native simulator WindowFeatures -> model features.
    fc_mean = float(d.get("force_Fc_mean", 0))
    fc_rms = float(d.get("force_Fc_rms", 0))
    fc_peak = float(d.get("force_Fc_peak", 0))

    # Simulator's current WindowFeatures currently contains Fc, not Ff/Fp.
    # These are explicit proxies for the existing model interface.
    ff_mean = float(d.get("force_Ff_mean", fc_mean))
    ff_rms = float(d.get("force_Ff_rms", fc_rms))
    ff_peak = float(d.get("force_Ff_peak", fc_peak))
    fp_mean = float(d.get("force_Fp_mean", fc_mean))
    fp_rms = float(d.get("force_Fp_rms", fc_rms))
    fp_peak = float(d.get("force_Fp_peak", fc_peak))

    vib_mean = float(d.get("vib_z_mean", 0))
    vib_rms = float(d.get("vib_z_rms", 0))
    ae_rms = float(d.get("ae_rms", 0))

    feed = float(d.get("feed_mm_per_rev", 0))
    speed = float(d.get("cuttingSpeed_m_per_min", 0))
    diameter = float(d.get("workpieceDiameter_mm", 50))
    rpm = speed * 1000.0 / (math.pi * diameter) if diameter > 0 else 0

    return {
        "F_f_RMS": ff_rms, "F_f_MEAN": ff_mean, "F_f_MAX": ff_peak,
        "F_c_RMS": fc_rms, "F_c_MEAN": fc_mean, "F_c_MAX": fc_peak,
        "F_p_RMS": fp_rms, "F_p_MEAN": fp_mean, "F_p_MAX": fp_peak,
        "CV3.Z_MEAN": vib_mean, "CV3.Z_RMS": vib_rms,
        "TV2.Z_MEAN": vib_mean, "TV2.Z_RMS": vib_rms,
        "AE_RMS": ae_rms, "AE_C_0.1": ae_rms,
        "CON.G.FREAL": feed, "CON.A.SREAL.S": rpm,
    }

def from_raw(frames, params=None):
    fc = [f.get("force_Fc_N", 0) for f in frames]
    ff = [f.get("force_Ff_N", 0) for f in frames]
    fp = [f.get("force_Fp_N", 0) for f in frames]
    vib = [f.get("vibration_Z_g", 0) for f in frames]
    ae = [f.get("acousticEmission_RMS_V", 0) for f in frames]

    ff_mean, ff_rms, ff_peak = stats(ff)
    fc_mean, fc_rms, fc_peak = stats(fc)
    fp_mean, fp_rms, fp_peak = stats(fp)
    vib_mean, vib_rms, _ = stats(vib)
    ae_mean, ae_rms, _ = stats(ae)

    params = params or {}
    feed = float(params.get("feed_mm_per_rev", 0))
    speed = float(params.get("cuttingSpeed_m_per_min", 0))
    diameter = float(params.get("workpieceDiameter_mm", 50))
    rpm = speed * 1000.0 / (math.pi * diameter) if diameter > 0 else 0

    return {
        "F_f_RMS": ff_rms, "F_f_MEAN": ff_mean, "F_f_MAX": ff_peak,
        "F_c_RMS": fc_rms, "F_c_MEAN": fc_mean, "F_c_MAX": fc_peak,
        "F_p_RMS": fp_rms, "F_p_MEAN": fp_mean, "F_p_MAX": fp_peak,
        "CV3.Z_MEAN": vib_mean, "CV3.Z_RMS": vib_rms,
        "TV2.Z_MEAN": vib_mean, "TV2.Z_RMS": vib_rms,
        "AE_RMS": ae_rms, "AE_C_0.1": ae_rms,
        "CON.G.FREAL": feed, "CON.A.SREAL.S": rpm,
    }

def predict(payload):
    if "frames" in payload:
        features = from_raw(payload["frames"], payload.get("params"))
    else:
        features = from_features(payload)

    X = pd.DataFrame([[features[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    wear = max(0.0, float(model.predict(X)[0]))

    return {
        "predicted_wear_mm": round(wear, 6),
        "condition": condition(wear),
        "simulation_time_sec": payload.get("simulation_time_sec"),
        "features": features
    }

class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"status": "ok"})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "running", "model": MODEL_FILE})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "Use POST /predict"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode())
            self.send_json(200, predict(payload))
        except Exception as e:
            self.send_json(500, {"error": str(e)})

if __name__ == "__main__":
    print("=" * 60)
    print("AI LATHE TOOL WEAR - LIVE ML ADAPTER")
    print("=" * 60)
    print(f"Model : {MODEL_FILE}")
    print(f"Port  : {PORT}")
    print("POST  : http://localhost:8765/predict")
    print("GET   : http://localhost:8765/health")
    print("Waiting for simulator data...")
    print("=" * 60)
    HTTPServer((HOST, PORT), Handler).serve_forever()
