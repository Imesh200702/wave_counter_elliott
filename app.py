import streamlit as st
import ccxt
import pandas as pd
import google.generativeai as genai
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# --- IMPORT KNOWLEDGE BASE ---
try:
    from knowledge import ELLIOTT_KNOWLEDGE
except ImportError:
    st.error("❌ 'knowledge.py' file not found! Please upload it to GitHub.")
    st.stop()

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(page_title="Elliott Wave Pro (1m-1W)", layout="wide", page_icon="🌊")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background-color: #1e2130; border: 1px solid #2b2f42;
        padding: 15px; border-radius: 8px; color: white; margin-bottom: 10px;
    }
    .deep-analysis-box {
        background-color: #151922; border-left: 4px solid #7c4dff;
        padding: 15px; border-radius: 5px; margin-bottom: 10px;
    }
    .scenario-card {
        background-color: #262730; padding: 20px; border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")

if not API_KEY:
    st.warning("⚠️ API Key Required. Add it to Streamlit Secrets or Sidebar.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    # CHANGED: Use a stable model for Cloud deployment
    MODEL_NAME = 'gemini-3-pro-preview' 
except Exception as e:
    st.error(f"API Configuration Error: {e}")
    st.stop()

# ==========================================
# 2. STATE & DATA FUNCTIONS
# ==========================================

if "ai_data" not in st.session_state: st.session_state.ai_data = None
if "df_micro" not in st.session_state: st.session_state.df_micro = None
if "last_update" not in st.session_state: st.session_state.last_update = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

def get_crypto_data(symbol, timeframe, limit=200):
    """
    Fetches crypto data. Tries Binance first.
    """
    try:
        # NOTE: If Binance blocks Streamlit Cloud IP, change this to ccxt.kraken() or ccxt.coinbase()
        exchange = ccxt.binance({'enableRateLimit': True}) 
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol} on {timeframe}: {e}")
        return pd.DataFrame()

def analyze_deep_wave(symbol, micro_tf, df_1w, df_1d, df_micro, language, previous=None):
    
    lang_inst = "Explain in English."
    if language == "Singlish":
        lang_inst = "Explain in 'Singlish' (Sinhala mixed with English). Use technical terms freely."

    # Data to JSON (limit candles to reduce token count)
    json_1w = df_1w.tail(50).to_json(orient="records", date_format='iso')
    json_1d = df_1d.tail(100).to_json(orient="records", date_format='iso')
    json_micro = df_micro.tail(100).to_json(orient="records", date_format='iso')

    task = "### TASK: DETAILED ELLIOTT WAVE STRUCTURE ANALYSIS"
    if previous: task += f" (UPDATE PREVIOUS ANALYSIS)"

    # Specific instruction for 1m/3m scalping
    scalp_instruction = ""
    if micro_tf in ['1m', '3m', '5m']:
        scalp_instruction = """
        **⚠️ SCALPING MODE ACTIVE (1m-5m):**
        - The Micro chart is very volatile.
        - Focus on immediate short-term structures.
        - Ensure the trade direction aligns with the 1D Trend.
        - Tight invalidation levels are required.
        """

    prompt = f"""
    You are a Master Elliott Wave Analyst.
    {task}
    
    ### DATA
    * **1W (Trend):** {json_1w}
    * **1D (Swing):** {json_1d}
    * **MICRO ({micro_tf}):** {json_micro}
    
    ### INSTRUCTIONS
    1.  **Structure:** Explain the hierarchy (e.g., "Micro Wave 3 of Minor Wave 5").
    2.  **Sub-Waves:** Identify internal sub-waves (i, ii, iii, iv, v).
    3.  **Volume:** Confirm count with volume.
    {scalp_instruction}
    
    ### LANGUAGE
    {lang_inst}
    
    ### OUTPUT JSON FORMAT (MANDATORY)
    {{
        "macro_analysis": {{
            "trend": "Bullish/Bearish",
            "current_structure": "Primary Wave Count",
            "detailed_breakdown": "History of the move.",
            "key_levels": "Key Support/Res"
        }},
        "micro_analysis": {{
            "timeframe": "{micro_tf}",
            "current_wave_degree": "Degree (e.g. Sub-Minuette)",
            "wave_count_status": "Status (e.g. Wave iii extending)",
            "sub_wave_structure": "Internal structure description.",
            "fib_confluence": "Fib levels aligned.",
            "volume_validation": "Volume analysis."
        }},
        "trade_scenarios": [
            {{
                "name": "Primary Setup",
                "trade_type": "Long/Short",
                "probability": "High",
                "summary": "Reasoning.",
                "entry_zone": "0.00 - 0.00",
                "target": 0.00,
                "invalidation": 0.00,
                "color": "#00E676"
            }},
            {{
                "name": "Alternative",
                "trade_type": "Long/Short",
                "probability": "Low",
                "summary": "Alt count.",
                "entry_zone": "0.00 - 0.00",
                "target": 0.00,
                "invalidation": 0.00,
                "color": "#FFAB00"
            }}
        ]
    }}
    """
    
    try:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=ELLIOTT_KNOWLEDGE)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        st.error(f"AI Analysis Error: {e}")
        return None

# ==========================================
# 3. UI LAYOUT
# ==========================================

with st.sidebar:
    st.title("🌊 Deep Wave AI")
    lang = st.radio("Language", ["Singlish", "English"], index=0)
    sym = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT"], index=0)
    
    # --- UPDATED TIMEFRAME LIST ---
    st.subheader("Timeframe")
    tf = st.selectbox(
        "Select Micro Chart", 
        ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"], 
        index=0 
    )
    
    run = st.button("🚀 Analyze Structure", type="primary")
    
    st.divider()
    with st.popover("💬 Chat about Structure"):
        q = st.chat_input("Ask about the wave count...")
        if q and st.session_state.ai_data:
            st.session_state.chat_history.append({"role": "user", "content": q})
            try:
                mod = genai.GenerativeModel(MODEL_NAME)
                # Pass shortened context to chat to avoid limits
                context_str = json.dumps(st.session_state.ai_data)
                res = mod.generate_content(f"Context: {context_str}. User: {q}. Lang: {lang}")
                st.session_state.chat_history.append({"role": "assistant", "content": res.text})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
        
        for m in st.session_state.chat_history: st.chat_message(m['role']).write(m['content'])
        
        if st.button("Clear Chat"): 
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# 4. MAIN PAGE
# ==========================================

st.title(f"🌊 {sym} Deep Analysis ({tf})")

if run:
    with st.spinner(f"📡 Fetching Data for {sym}..."):
        # Fetching data
        d1w = get_crypto_data(sym, "1w", 100)
        d1d = get_crypto_data(sym, "1d", 150)
        dm = get_crypto_data(sym, tf, 150)
        
        # --- ERROR HANDLING: Check if data is empty ---
        if not dm.empty and not d1w.empty:
            st.session_state.df_micro = dm
            st.session_state.df_1w = d1w
            st.session_state.df_1d = d1d
            
            with st.spinner("🧠 AI Analyzing Elliott Waves..."):
                ai = analyze_deep_wave(sym, tf, d1w, d1d, dm, lang)
                if ai:
                    st.session_state.ai_data = ai
                    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history = []
                else:
                    st.error("❌ AI Analysis Failed. Please try again.")
        else:
            st.error(f"❌ Failed to fetch data for {sym}.")
            st.warning("⚠️ Probable Cause: Binance API is blocking the Streamlit Cloud IP.")
            st.info("💡 Tip: If you are the developer, try changing `ccxt.binance()` to `ccxt.kraken()` or `ccxt.coinbase()` in the code.")

# --- DISPLAY RESULTS ---
if st.session_state.ai_data:
    data = st.session_state.ai_data
    macro = data.get('macro_analysis', {})
    micro = data.get('micro_analysis', {})
    
    # Update Button
    c1, c2 = st.columns([5,1])
    c1.caption(f"Last Update: {st.session_state.last_update}")
    
    # --- 1. MACRO BREAKDOWN ---
    st.markdown("### 🌍 Macro Structure (1W / 1D)")
    st.markdown(f"""
    <div class='metric-card'>
        <h3>{macro.get('trend')} - {macro.get('current_structure')}</h3>
        <p style='color:#bbb;'>{macro.get('detailed_breakdown')}</p>
        <small>🔑 Key Levels: {macro.get('key_levels')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # --- 2. MICRO DEEP DIVE ---
    st.markdown(f"### 🔬 Micro Wave Analysis ({tf})")
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class='deep-analysis-box'>
            <b>🌊 Current Wave Degree:</b> {micro.get('current_wave_degree')}<br>
            <b>📍 Status:</b> {micro.get('wave_count_status')}
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class='deep-analysis-box'>
            <b>📐 Fib Confluence:</b> {micro.get('fib_confluence')}<br>
            <b>📊 Volume:</b> {micro.get('volume_validation')}
        </div>
        """, unsafe_allow_html=True)

    st.info(f"**📝 Sub-Wave Structure:** {micro.get('sub_wave_structure')}")

    # --- 3. CHART & SCENARIOS ---
    tab1, tab2 = st.tabs(["📈 Chart (Price + Volume)", "🛠 Trade Setup"])
    
    with tab1:
        if st.session_state.df_micro is not None:
            df = st.session_state.df_micro
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            colors = ['#00E676' if r['close'] >= r['open'] else '#FF5252' for i, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            
            # Add Levels if available
            if 'trade_scenarios' in data and len(data['trade_scenarios']) > 0:
                prim = data['trade_scenarios'][0]
                try:
                    fig.add_hline(y=float(prim['target']), line_dash="dash", line_color="#00E676", row=1, col=1, annotation_text="TP")
                    fig.add_hline(y=float(prim['invalidation']), line_dash="dot", line_color="#FF5252", row=1, col=1, annotation_text="Invalidation")
                except: pass
            
            fig.update_layout(height=600, template="plotly_dark", title=f"{sym} {tf} Analysis", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if 'trade_scenarios' in data:
            for s in data['trade_scenarios']:
                st.markdown(f"""
                <div class='scenario-card' style='border-left: 5px solid {s.get('color')};'>
                    <h3>{s.get('name')} ({s.get('trade_type')})</h3>
                    <p>{s.get('summary')}</p>
                    <div style='background:#111; padding:10px; border-radius:5px;'>
                        🔵 Entry: {s.get('entry_zone')}<br>
                        🟢 Target: {s.get('target')}<br>
                        🔴 Invalidation: {s.get('invalidation')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
