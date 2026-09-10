
# Simulator -> Existing ML Model Connection

## What this does
The React CNC lathe simulator remains the INPUT source.
It sends its rolling virtual sensor window to `live_adapter.py`.
The adapter calls the EXISTING `sensor_tool_condition_model.pkl`.
The prediction is shown in the simulator UI.

## Important
The adapter uses explicit proxy mappings for channels that the original
experimental dataset has but the simulator does not reproduce exactly.
This proves the end-to-end integration; it is not a claim that those
channels are physically identical.

## Run
1. Put `live_adapter.py` in `E:\AI_Tool_Wear_Prediction` beside:
   `sensor_tool_condition_model.pkl`
2. Stop the old adapter with Ctrl+C.
3. Replace it with this adapter and run:
   `py live_adapter.py`
4. Start the simulator with its normal Vite/Bun command.
5. Start/play the simulation.
6. The simulator will POST a rolling sensor window every ~0.5 s to:
   `http://localhost:8765/predict`
7. The AI prediction appears in the simulator under "AI Tool Wear Prediction".

## If the browser says CORS/network error
Make sure `py live_adapter.py` is running first. The adapter includes CORS headers.
