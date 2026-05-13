import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title=" Moneyball Scout ", page_icon="⚽", layout="wide")

# --- 1. HELPER: CLEAN POSITIONS ---
def get_position_group(pos):
    """Maps specific positions (ST, LW, etc.) to broad groups (Attacker, etc.)"""
    if pd.isna(pos): return "Unknown"
    
    # Handle composite positions (e.g., "FW,MF" -> takes "FW")
    main_pos = str(pos).split(',')[0].upper().strip()
    
    if main_pos in ['ST', 'LW', 'RW', 'FW', 'CF', 'LS', 'RS']: return 'Attacker'
    if main_pos in ['MF', 'DM', 'AM', 'CM', 'LM', 'RM']: return 'Midfielder'
    if main_pos in ['CB', 'LB', 'RB', 'DF', 'DEF', 'LWB', 'RWB']: return 'Defender'
    if main_pos in ['GK']: return 'Goalkeeper'
    return 'Other'

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    # Load the file
    df = pd.read_csv("players.csv")
    
    # Clean duplicates & reset index
    df = df.drop_duplicates(subset=['name']).reset_index(drop=True)
    
    # Handle Pakistani players (missing league/club)
    df['league'] = df['league'].fillna('Pakistan Premier League')
    df['club'] = df['club'].fillna('Unattached / Domestic')
    
    # Create the "Broad Position" column for the filter
    df['position_group'] = df['position'].apply(get_position_group)
    
    # Fill numeric NAs with 0
    df = df.fillna(0)
    
    return df

df = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("🕵️‍♂️ Scout Filters")

# FILTER 1: Position Group (Solves the ST vs FW confusion)
position_groups = ["All", "Attacker", "Midfielder", "Defender", "Goalkeeper"]
selected_group = st.sidebar.selectbox("Position Group:", position_groups)

# Filter the dataset based on group
if selected_group != "All":
    filtered_df = df[df['position_group'] == selected_group]
else:
    filtered_df = df

# FILTER 2: Target Player
# Only show players from the selected group in the dropdown
players_list = sorted(filtered_df['name'].unique())

# Try to default to Messi if he's in the list, otherwise pick the first one
default_index = players_list.index("Lionel Messi") if "Lionel Messi" in players_list else 0
selected_player = st.sidebar.selectbox("Select Target Player:", players_list, index=default_index)

# FILTER 3: Strategy
st.sidebar.divider()
st.sidebar.caption("Scouting Strategy")
local_talent = st.sidebar.checkbox("🇵🇰 Search Pakistani Talent Only", value=True)

# --- 4. THE AI ENGINE ---
feature_cols = ['xG', 'xA', 'passes_per90', 'pass_accuracy_pct', 
                'dribbles_per90', 'tackles_per90', 'interceptions_per90']

# Normalize data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

# Define the Pool (Global vs Local)
if local_talent:
    candidate_pool = df_normalized[df_normalized['league'] == 'Pakistan Premier League']
else:
    candidate_pool = df_normalized

# Train KNN
knn = NearestNeighbors(n_neighbors=6, algorithm='auto')
knn.fit(candidate_pool[feature_cols])

# Get Target Stats
target_row = df_normalized[df_normalized['name'] == selected_player]
if target_row.empty:
    st.error("Player not found in current filters. Please adjust the Position Group.")
    st.stop()
    
target_stats = target_row[feature_cols].values
target_val = target_row.iloc[0]['market_value_eur']

# Find Neighbors
distances, indices = knn.kneighbors(target_stats)

# --- 5. UI: HERO SECTION ---
st.title(f"Moneyball Scout: {selected_player}")
st.caption("Using Machine Learning to find statistically similar players at a lower price.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Position", target_row.iloc[0]['position'])
col2.metric("Market Value", f"€{target_val:,}")
col3.metric("League", target_row.iloc[0]['league'])
col4.metric("Age", target_row.iloc[0]['age'])

st.divider()

# --- 6. RECOMMENDATIONS ---
st.subheader(f"⚡ Recommended Alternatives ({'🇵🇰 Local' if local_talent else '🌍 Global'})")

# Prepare list of recommended indices (skipping self if in list)
rec_indices = [i for i in indices[0] if candidate_pool.iloc[i]['name'] != selected_player][:5]

# Display Cards
cols = st.columns(5)
rec_players_data = [] # Store for the radar chart later

for i, idx in enumerate(rec_indices):
    rec_row = candidate_pool.iloc[idx]
    rec_players_data.append(rec_row) # Save for dropdown
    
    savings = target_val - rec_row['market_value_eur']
    
    with cols[i]:
        st.markdown(f"#### {rec_row['name']}")
        st.caption(f"{rec_row['club']}")
        st.write(f"**€{rec_row['market_value_eur']:,}**")
        
        if savings > 0:
            st.markdown(f":green[Save €{savings:,}]")
        else:
            st.markdown(f":red[+€{abs(savings):,}]")

# --- 7. INTERACTIVE COMPARISON ---
st.divider()

# Create tabs for better organization
tab1, tab2 = st.tabs(["📊 Head-to-Head Comparison", "🔢 Detailed Stats"])

with tab1:
    st.subheader("Visual Comparison")
    
    # DROPDOWN: Select which recommended player to compare
    rec_names = [p['name'] for p in rec_players_data]
    compare_name = st.selectbox("Select player to compare with Target:", rec_names)
    
    # Get stats for the selected comparison player
    comp_row = next(p for p in rec_players_data if p['name'] == compare_name)
    
    # Radar Chart
    fig = go.Figure()
    
    # Trace 1: Target Player
    fig.add_trace(go.Scatterpolar(
        r=target_row.iloc[0][feature_cols].values,
        theta=feature_cols,
        fill='toself',
        name=selected_player,
        line_color='blue',
        opacity=0.7
    ))
    
    # Trace 2: Comparison Player
    fig.add_trace(go.Scatterpolar(
        r=comp_row[feature_cols].values,
        theta=feature_cols,
        fill='toself',
        name=comp_row['name'],
        line_color='green',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Raw Data View")
    st.dataframe(pd.DataFrame([target_row.iloc[0]] + rec_players_data)[['name', 'position', 'market_value_eur'] + feature_cols])