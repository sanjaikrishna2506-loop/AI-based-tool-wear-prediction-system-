import json, math
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

MODEL_FILE = 'sensor_tool_condition_model.pkl'
HOST, PORT = '0.0.0.0', 8765

FEATURES = [
 'F_f_RMS','F_f_MEAN','F_f_MAX','F_c_RMS','F_c_MEAN','F_c_MAX',
 'F_p_RMS','F_p_MEAN','F_p_MAX','CV3.Z_MEAN','CV3.Z_RMS',
 'TV2.Z_MEAN','TV2.Z_RMS','AE_RMS','AE_C_0.1','CON.G.FREAL','CON.A.SREAL.S'
]

# Feature ranges observed in the experimental training dataset.
TRAIN_RANGES = {
 'F_f': (274.75, 2044.93), 'F_c': (480.58, 1058.27), 'F_p': (101.89, 905.89),
 'CV3Z': (9.796, 29.664), 'TV2Z': (-106.856, 200.868),
 'AE_RMS': (0.003175, 0.152073), 'AE_C01': (0.0, 173293.0),
 'FEED': (0.0, 84091.87798), 'SPEED': (6906771.019, 12371794.46)
}

def stats(vals):
    vals = [float(v) for v in vals]
    if not vals: return 0.0, 0.0, 0.0
    mean = sum(vals)/len(vals)
    rms = math.sqrt(sum(v*v for v in vals)/len(vals))
    peak = max(abs(v) for v in vals)
    return mean, rms, peak

def scale(v, src_lo, src_hi, dst_lo, dst_hi):
    if src_hi == src_lo: return (dst_lo+dst_hi)/2
    x = (float(v)-src_lo)/(src_hi-src_lo)
    x = max(0.0, min(1.0, x))
    return dst_lo + x*(dst_hi-dst_lo)

def condition(w):
    if w < 0.15: return 'HEALTHY'
    if w < 0.20: return 'MONITOR'
    if w < 0.25: return 'WARNING'
    return 'TOOL CHANGE'

def build_features(payload):
    frames = payload.get('frames', [])
    if not frames:
        raise ValueError('No simulator frames received')

    fc = [f.get('force_Fc_N',0) for f in frames]
    ff = [f.get('force_Ff_N',0) for f in frames]
    fp = [f.get('force_Fp_N',0) for f in frames]
    vz = [f.get('vibration_Z_g',0) for f in frames]
    ae = [f.get('acousticEmission_RMS_V',0) for f in frames]

    ff_m, ff_r, ff_p = stats(ff)
    fc_m, fc_r, fc_p = stats(fc)
    fp_m, fp_r, fp_p = stats(fp)
    vz_m, vz_r, _ = stats(vz)
    ae_m, ae_r, ae_p = stats(ae)

    p = payload.get('params', {})
    feed = float(p.get('feed_mm_per_rev', 0.25))
    speed = float(p.get('cuttingSpeed_m_per_min', 180.0))
    dia = float(p.get('workpieceDiameter_mm', 50.0))
    rpm = speed*1000/(math.pi*dia) if dia > 0 else 0

    # Calibrate simulator-native physical signals into the numerical feature
    # space used by the existing experimental ML model. This does NOT retrain
    # the model and does not add simulated wear to the prediction.
    # Source envelopes are deliberately based on the simulator's control/signal
    # ranges, while destination envelopes come from the training dataset.
    Ff = [scale(x, 300, 2200, *TRAIN_RANGES['F_f']) for x in (ff_m, ff_r, ff_p)]
    Fc = [scale(x, 500, 3000, *TRAIN_RANGES['F_c']) for x in (fc_m, fc_r, fc_p)]
    Fp = [scale(x, 0, 100, *TRAIN_RANGES['F_p']) for x in (fp_m, fp_r, fp_p)]
    VZ_m = scale(vz_m, 0.0, 1.0, *TRAIN_RANGES['CV3Z'])
    VZ_r = scale(vz_r, 0.0, 1.0, *TRAIN_RANGES['CV3Z'])
    TVZ_m = scale(vz_m, 0.0, 1.0, *TRAIN_RANGES['TV2Z'])
    TVZ_r = scale(vz_r, 0.0, 1.0, *TRAIN_RANGES['TV2Z'])
    AEr = scale(ae_r, 0.0, 2.0, *TRAIN_RANGES['AE_RMS'])
    AEc = scale(ae_p, 0.0, 2.0, *TRAIN_RANGES['AE_C01'])
    Feed = scale(feed, 0.10, 0.50, *TRAIN_RANGES['FEED'])
    Speed = scale(speed, 80, 350, *TRAIN_RANGES['SPEED'])

    return {
      'F_f_RMS':Ff[1], 'F_f_MEAN':Ff[0], 'F_f_MAX':Ff[2],
      'F_c_RMS':Fc[1], 'F_c_MEAN':Fc[0], 'F_c_MAX':Fc[2],
      'F_p_RMS':Fp[1], 'F_p_MEAN':Fp[0], 'F_p_MAX':Fp[2],
      'CV3.Z_MEAN':VZ_m, 'CV3.Z_RMS':VZ_r,
      'TV2.Z_MEAN':TVZ_m, 'TV2.Z_RMS':TVZ_r,
      'AE_RMS':AEr, 'AE_C_0.1':AEc,
      'CON.G.FREAL':Feed, 'CON.A.SREAL.S':Speed,
    }

model = joblib.load(MODEL_FILE)

class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
        self.end_headers(); self.wfile.write(body)
    def do_OPTIONS(self): self.send_json(200, {'status':'ok'})
    def do_GET(self):
        self.send_json(200, {'status':'running','model':MODEL_FILE}) if self.path=='/health' else self.send_json(404, {'error':'not found'})
    def do_POST(self):
        if self.path != '/predict': return self.send_json(404, {'error':'Use POST /predict'})
        try:
            n=int(self.headers.get('Content-Length',0)); payload=json.loads(self.rfile.read(n))
            features=build_features(payload)
            X=pd.DataFrame([[features[c] for c in FEATURES]], columns=FEATURES)
            wear=max(0.0,float(model.predict(X)[0]))
            self.send_json(200, {'predicted_wear_mm':round(wear,6),'condition':condition(wear), 'simulation_time_sec':payload.get('simulation_time_sec')})
        except Exception as e:
            self.send_json(500, {'error':str(e)})

if __name__=='__main__':
    print('='*60); print('AI LATHE TOOL WEAR - LIVE ML ADAPTER'); print('='*60)
    print('Model :',MODEL_FILE); print('Host  :',HOST); print('Port  :',PORT)
    print('Prediction endpoint: http://localhost:8765/predict')
    print('Health check:       http://localhost:8765/health')
    print('Waiting for simulator data...'); print('='*60)
    HTTPServer((HOST,PORT),Handler).serve_forever()
