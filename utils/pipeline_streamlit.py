# pipeline.py
import utils.data_utils as du
import pandas as pd
import streamlit as st

def preprocess_pipeline(df: pd.DataFrame,
                        drop_short_cycles: bool = False):
    """
    1. Check structure → ValueError if there are missing columns
    2. Analyze cycles → Return too_short, too_long
    3. If drop_short_cycles=True → Delete short cycles
       If False but too_short exists → Don't continue
    4. Always trim_cycles if too_long is found
    5. clean_columns + scale_features → Return df_ready
    """
    # First clean column names to validate structure
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    ok, missing = du.validate_structure(df)
    if not ok:
        raise ValueError(f"Missing columns: {missing}")

    # Always remove rows with Cycle Loop 0, regardless of mode
    if "Cycle Loop" in df.columns:
        df = df.loc[df["Cycle Loop"] != 0]
        
    # Also remove rows with Mode "Resting" if Mode column exists
    if "Mode" in df.columns:
        df = df.loc[df["Mode"].str.strip() != "Resting"]
    
    too_short, too_long = du.analyze_cycles(df)
    df2 = df
    
    if too_short:
        if drop_short_cycles:
            df2 = du.drop_cycles(df2, list(too_short.keys()))
        else:
            # Let the caller decide first
            return None, too_short, too_long
    
    if too_long:
        df2 = du.trim_cycles(df2)
    
    df3 = du.clean_columns(df2)
    df4 = du.scale_features(df3)
    return df4, too_short, too_long

def handle_pipeline(df_raw):
    """
    Call preprocess_pipeline, handle short/long cycles
    Return df_processed or st.stop() if canceled
    """
    # First pipeline call
    try:
        result, too_short, too_long = preprocess_pipeline(
            df_raw, drop_short_cycles=False
        )
    except ValueError as e:
        st.error(f"Invalid structure! {e}")
        st.stop()
    
    # If short cycles exist
    if too_short:
        st.warning(f"Found cycle loops with row < 60: {too_short}")
        drop = st.checkbox("Delete these cycle loops and continue", value=False)
        if not drop:
            st.error("Short cycle loops not deleted, pipeline canceled")
            st.stop()
        # Call again with drop_short_cycles=True
        result, too_short2, too_long2 = preprocess_pipeline(
            df_raw, drop_short_cycles=True
        )
        too_long = too_long2
    else:
        # no short cycles
        too_long = too_long if 'too_long' in locals() else {}
    
    # If long cycles exist, they'll be automatically trimmed
    if too_long:
        st.info(f"Found cycle loops with row > 60 and they were trimmed: {too_long}")
    
    return result