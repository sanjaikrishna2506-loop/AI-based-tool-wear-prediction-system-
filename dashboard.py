import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Lathe Tool Condition Monitor",
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "sensor_tool_condition_model.pkl"
DATA_FILE = "Dataset_Machine_tool_wear_mm.csv"

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"Could not load model: {MODEL_FILE}")
    st.error(e)
    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Could not load dataset: {DATA_FILE}")
    st.error(e)
    st.stop()


# Remove columns that contained missing values during training
for col in ["CV3.X_SK", "CV3.X_KURT"]:
    if col in df.columns:
        df = df.drop(columns=[col])


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    "F_f_RMS",
    "F_f_MEAN",
    "F_f_MAX",

    "F_c_RMS",
    "F_c_MEAN",
    "F_c_MAX",

    "F_p_RMS",
    "F_p_MEAN",
    "F_p_MAX",

    "CV3.Z_MEAN",
    "CV3.Z_RMS",

    "TV2.Z_MEAN",
    "TV2.Z_RMS",

    "AE_RMS",
    "AE_C_0.1",

    "CON.G.FREAL",
    "CON.A.SREAL.S"
]


# ============================================================
# TOOL CONDITION FUNCTION
# ============================================================

def get_condition(wear):

    if wear < 0.15:
        return "HEALTHY"

    elif wear < 0.20:
        return "MONITOR"

    elif wear < 0.25:
        return "WARNING"

    else:
        return "TOOL CHANGE"


# ============================================================
# HEADER
# ============================================================

st.title("🔧 AI-Based Lathe Tool Condition Monitoring")

st.write(
    "Sensor-based prediction of flank wear and tool condition "
    "using machine and machining signals."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("System")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Dataset Replay",
        "Manual Demo"
    ]
)


# ============================================================
# DATASET REPLAY MODE
# ============================================================

if mode == "Dataset Replay":

    st.subheader("📡 Recorded Lathe Sensor Replay")

    st.info(
        "This mode replays experimentally recorded lathe sensor data "
        "and sends each selected observation to the trained ML model."
    )

    # Only use tools that have data
    available_tools = sorted(df["tool"].unique())

    selected_tool = st.selectbox(
        "Select Tool",
        available_tools
    )

    # Select data for the chosen tool
    tool_data = df[df["tool"] == selected_tool].reset_index(drop=True)

    max_index = len(tool_data) - 1

    row_number = st.slider(
        "Machining Progress",
        min_value=0,
        max_value=max_index,
        value=0,
        help="Move through the recorded machining data."
    )

    current_row = tool_data.iloc[row_number]

    # ========================================================
    # PREDICTION
    # ========================================================

    input_data = pd.DataFrame(
        [[current_row[feature] for feature in features]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    # Prevent negative predictions
    prediction = max(0.0, prediction)

    condition = get_condition(prediction)


    # ========================================================
    # MACHINE INFORMATION
    # ========================================================

    st.subheader("Machine Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Tool ID",
            str(selected_tool)
        )

    with col2:
        st.metric(
            "Recorded Sample",
            f"{row_number + 1} / {len(tool_data)}"
        )

    with col3:
        st.metric(
            "Machining Time",
            f"{row_number} sec"
        )


    # ========================================================
    # SENSOR INFORMATION
    # ========================================================

    st.subheader("Sensor / Machine Data")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("### Feed Force")

        st.metric(
            "RMS",
            f"{current_row['F_f_RMS']:.3f}"
        )

        st.metric(
            "Mean",
            f"{current_row['F_f_MEAN']:.3f}"
        )

        st.metric(
            "Maximum",
            f"{current_row['F_f_MAX']:.3f}"
        )


    with col2:

        st.write("### Cutting Force")

        st.metric(
            "RMS",
            f"{current_row['F_c_RMS']:.3f}"
        )

        st.metric(
            "Mean",
            f"{current_row['F_c_MEAN']:.3f}"
        )

        st.metric(
            "Maximum",
            f"{current_row['F_c_MAX']:.3f}"
        )


    with col3:

        st.write("### Other Signals")

        st.metric(
            "Vibration RMS",
            f"{current_row['CV3.Z_RMS']:.3f}"
        )

        st.metric(
            "AE RMS",
            f"{current_row['AE_RMS']:.3f}"
        )

        st.metric(
            "CNC Feed Signal",
            f"{current_row['CON.G.FREAL']:.2f}"
        )


    # ========================================================
    # AI RESULT
    # ========================================================

    st.divider()

    st.subheader("🤖 AI Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Flank Wear",
            f"{prediction:.4f} mm"
        )

        st.write(
            "Wear Limit: **0.2500 mm**"
        )

    with col2:

        if condition == "HEALTHY":

            st.success(
                "🟢 TOOL CONDITION: HEALTHY"
            )

        elif condition == "MONITOR":

            st.warning(
                "🟡 TOOL CONDITION: MONITOR"
            )

        elif condition == "WARNING":

            st.warning(
                "🟠 TOOL CONDITION: WARNING"
            )

        else:

            st.error(
                "🔴 TOOL CONDITION: TOOL CHANGE"
            )


    # ========================================================
    # WEAR TREND
    # ========================================================

    st.divider()

    st.subheader("📈 Predicted Wear Trend")

    # Predict all previous rows for the selected tool
    history = tool_data.iloc[:row_number + 1]

    history_inputs = history[features]

    history_predictions = model.predict(history_inputs)

    history_predictions = [
        max(0.0, value)
        for value in history_predictions
    ]

    chart_data = pd.DataFrame(
        {
            "Predicted Flank Wear (mm)": history_predictions
        }
    )

    st.line_chart(chart_data)

    st.caption(
        "The graph shows the ML-predicted flank wear as the "
        "recorded machining process progresses."
    )


# ============================================================
# MANUAL DEMO MODE
# ============================================================

else:

    st.subheader("🧪 Manual Sensor Demo")

    st.info(
        "Enter sensor values manually to demonstrate how the trained "
        "ML model converts machine signals into tool condition."
    )


    # --------------------------------------------------------
    # FORCE INPUTS
    # --------------------------------------------------------

    st.write("### Feed Force")

    col1, col2, col3 = st.columns(3)

    with col1:
        ff_rms = st.number_input(
            "F_f_RMS",
            value=100.0
        )

    with col2:
        ff_mean = st.number_input(
            "F_f_MEAN",
            value=100.0
        )

    with col3:
        ff_max = st.number_input(
            "F_f_MAX",
            value=100.0
        )


    st.write("### Cutting Force")

    col1, col2, col3 = st.columns(3)

    with col1:
        fc_rms = st.number_input(
            "F_c_RMS",
            value=100.0
        )

    with col2:
        fc_mean = st.number_input(
            "F_c_MEAN",
            value=100.0
        )

    with col3:
        fc_max = st.number_input(
            "F_c_MAX",
            value=100.0
        )


    st.write("### Other Force Signals")

    col1, col2, col3 = st.columns(3)

    with col1:
        fp_rms = st.number_input(
            "F_p_RMS",
            value=100.0
        )

    with col2:
        fp_mean = st.number_input(
            "F_p_MEAN",
            value=100.0
        )

    with col3:
        fp_max = st.number_input(
            "F_p_MAX",
            value=100.0
        )


    # --------------------------------------------------------
    # VIBRATION
    # --------------------------------------------------------

    st.write("### Vibration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cv3_z_mean = st.number_input(
            "CV3.Z_MEAN",
            value=0.0
        )

    with col2:
        cv3_z_rms = st.number_input(
            "CV3.Z_RMS",
            value=0.0
        )

    with col3:
        tv2_z_mean = st.number_input(
            "TV2.Z_MEAN",
            value=0.0
        )

    with col4:
        tv2_z_rms = st.number_input(
            "TV2.Z_RMS",
            value=0.0
        )


    # --------------------------------------------------------
    # ACOUSTIC EMISSION
    # --------------------------------------------------------

    st.write("### Acoustic Emission")

    col1, col2 = st.columns(2)

    with col1:
        ae_rms = st.number_input(
            "AE_RMS",
            value=0.0
        )

    with col2:
        ae_c01 = st.number_input(
            "AE_C_0.1",
            value=0.0
        )


    # --------------------------------------------------------
    # CNC SIGNALS
    # --------------------------------------------------------

    st.write("### CNC Signals")

    col1, col2 = st.columns(2)

    with col1:
        cnc_feed = st.number_input(
            "CON.G.FREAL",
            value=0.0
        )

    with col2:
        cnc_speed = st.number_input(
            "CON.A.SREAL.S",
            value=0.0
        )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Predict Tool Condition",
        use_container_width=True
    ):

        values = [[
            ff_rms,
            ff_mean,
            ff_max,

            fc_rms,
            fc_mean,
            fc_max,

            fp_rms,
            fp_mean,
            fp_max,

            cv3_z_mean,
            cv3_z_rms,

            tv2_z_mean,
            tv2_z_rms,

            ae_rms,
            ae_c01,

            cnc_feed,
            cnc_speed
        ]]

        input_data = pd.DataFrame(
            values,
            columns=features
        )

        prediction = model.predict(input_data)[0]

        prediction = max(0.0, prediction)

        condition = get_condition(prediction)


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("🤖 AI Result")

        st.metric(
            "Predicted Flank Wear",
            f"{prediction:.4f} mm"
        )

        st.write(
            "Wear Limit: **0.2500 mm**"
        )

        if condition == "HEALTHY":

            st.success(
                "🟢 TOOL CONDITION: HEALTHY"
            )

        elif condition == "MONITOR":

            st.warning(
                "🟡 TOOL CONDITION: MONITOR"
            )

        elif condition == "WARNING":

            st.warning(
                "🟠 TOOL CONDITION: WARNING"
            )

        else:

            st.error(
                "🔴 TOOL CONDITION: TOOL CHANGE"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Lathe Tool Condition Monitoring | "
    "Sensor-based flank wear prediction"
)