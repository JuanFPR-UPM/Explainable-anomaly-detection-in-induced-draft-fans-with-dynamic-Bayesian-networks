import altair as alt
import pickle
import time
import streamlit as st
import datetime
from pandas.errors import SettingWithCopyWarning
import pandas as pd
import warnings
import re
import numpy as np
from plotly_calplot import calplot
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri
pandas2ri.activate()
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
st.set_page_config(layout="wide")

# The object that represents the R session
r = robj.r

dbn_size = 6
# Fan defaults to IDF1
if 'fan' not in st.session_state:
    st.session_state.fan = 1
if 'progress_idx' not in st.session_state:
    st.session_state.progress_idx = 2

if 'lglk_threshold_history' not in st.session_state:
    st.session_state.lglk_threshold_history = -2600

if 'play' not in st.session_state:
    st.session_state.play = False

if st.session_state.fan == 1:

    r(f'''
    library(dbnR)
    load(file="size6_dmmhc_mmpc_hc_mi-g_bic-g_5e-06_fit.RDS")
    ''')

else:

    r(f'''
    library(dbnR)
    load(file="size6_dmmhc_mmpc_hc_mi-g_bic-g_5e-04_fit.RDS")
    ''')

# Load and cache files based on fan ID
@st.cache_data
def setup(fan):
    if fan == 1:
        from preprocessing import dfCV1, translation_dict
        dfCV = dfCV1
        anomaly_df = pd.read_excel(
            'Notas Corretivas IDF1.xlsx', sheet_name='ENG')
    else:
        from preprocessing import dfCV2, translation_dict
        dfCV = dfCV2
        anomaly_df = pd.read_excel(
            'Notas Corretivas IDF2.xlsx', sheet_name='ENG')

    anomaly_df.iloc[:, 8] = pd.to_datetime(
        anomaly_df.iloc[:, 8], format='%Y-%m-%d')
    anomalies = anomaly_df.iloc[:, 8]

    # Decode column names
    dfCV.columns = [translation_dict[x] for x in dfCV.columns]

    # Load precomputed history if exists
    try:
        history_lglk = np.load(f'history_lglk{fan}.npy')
    except OSError as e:
        with (robj.default_converter + pandas2ri.converter).context():
            robj.globalenv['dt_history'] = robj.conversion.get_conversion().py2rpy(
                dfCV.reset_index(drop=True))

        r(f'''
            f_dt_history <- fold_dt(dt_history, {dbn_size})
            lglk_history <- logLik(dbn, f_dt_history, by.sample=TRUE, na.rm=TRUE)
            ''')

        with (robj.default_converter + pandas2ri.converter).context():
            history_lglk = np.array(robj.conversion.get_conversion().rpy2py(
                robj.globalenv['lglk_history']))
        np.save(f'history_lglk{fan}',  history_lglk)

    return anomalies, dfCV, history_lglk

# Need to parameterize by "fan" so cache depends on it
@st.cache_data
def get_history_flags(lglk_threshold, fan):
    logl_df = pd.DataFrame(
        st.session_state.history_lglk.copy(), columns=['logl'])
    # Binarize based on threshold
    logl_df.loc[logl_df['logl'] >= lglk_threshold, 'logl'] = 0
    logl_df.loc[logl_df['logl'] < lglk_threshold, 'logl'] = 1
    logl_df.index = dfCV.iloc[dbn_size-1:, :].index
    logl_df = logl_df.groupby(by=logl_df.index.date, as_index=True).sum()
    return logl_df


anomalies, dfCV, st.session_state.history_lglk = setup(st.session_state.fan)

# Likelihoods are computed for the entire day
def eval_anomaly(start_day):
    end_day = start_day + datetime.timedelta(days=1)
    slice_df = dfCV[dfCV.index.to_series().between(start_day, end_day)]

    with (robj.default_converter + pandas2ri.converter).context():
        robj.globalenv['dt_val'] = robj.conversion.get_conversion().py2rpy(
            slice_df.reset_index(drop=True))

    r(f'''
    f_dt_val <- fold_dt(dt_val, {dbn_size})
    lglk <- logLik(dbn, f_dt_val, by.sample=TRUE, na.rm=TRUE)
    decomp_loglks <- c()
    for (node in names(f_dt_val)) {'{'}
        decomp_loglks <- cbind(decomp_loglks, logLik(dbn, f_dt_val, by.sample=TRUE, na.rm=TRUE, by.node=TRUE, nodes=node))
    {'}'}
    ''')

    with (robj.default_converter + pandas2ri.converter).context():
        logl = robj.conversion.get_conversion().rpy2py(robj.globalenv['lglk'])
        decomp_loglks = np.array(robj.conversion.get_conversion().rpy2py(
            robj.globalenv['decomp_loglks']))
        colnames = np.array(
            robj.conversion.get_conversion().rpy2py(r('names(f_dt_val)')))

    decomp_loglks = pd.DataFrame(decomp_loglks, columns=colnames)

    return logl, decomp_loglks, slice_df


# Adds a time-step each call
def plot_logl(chart, logl_df, col):
    with col:
        chart.add_rows(
            logl_df.iloc[st.session_state.progress_idx:st.session_state.progress_idx+1])

# Add time-steps both for the likelihood and raw data
def plot_feature(chart2, chart3, decomp_loglks, slice_df, col):
    with col:
        slice_decomp_logl = decomp_loglks.iloc[st.session_state.progress_idx:st.session_state.progress_idx +
                                               1, decomp_loglks.columns == st.session_state.feature]
        slice_feature = slice_df.iloc[st.session_state.progress_idx:st.session_state.progress_idx +
                                      dbn_size, slice_df.columns == st.session_state.feature]
        slice_decomp_logl.columns, slice_feature.columns = [
            st.session_state.feature.replace('.', '')], [st.session_state.feature.replace('.', '')]
        chart2.add_rows(slice_decomp_logl)
        chart3.add_rows(slice_feature)


def plot_ranking(decomp_loglks, anomaly_fires_idx, agg_window, placeholder):
    # If running, plot fired time-steps along the last 10 minutes
    if st.session_state.play:
        decomp_loglks = decomp_loglks.iloc[anomaly_fires_idx[np.where(np.logical_and(
            anomaly_fires_idx > st.session_state.progress_idx-dbn_size, anomaly_fires_idx <= st.session_state.progress_idx))[0]], :]
    # If paused index by agg_window and indices that were fired
    else:
        win_low, win_high = agg_window
        win_low, win_high = np.where(decomp_loglks.index.time == win_low)[
            0] - 1, np.where(decomp_loglks.index.time == win_high)[0]
        decomp_loglks = decomp_loglks.iloc[anomaly_fires_idx[np.where(np.logical_and(
            anomaly_fires_idx > win_low, anomaly_fires_idx <= win_high))[0]], :]

    decomp_sloglks = pd.DataFrame(decomp_loglks.sum(axis=0)).T

    # Sort descending and select top k
    sorted_df = decomp_sloglks.sort_values(
        by=0, axis=1, ascending=True).iloc[:, :st.session_state.k]
    sorted_df = sorted_df.T
    sorted_df.columns = ['contribution']
    # Replace former plot
    with placeholder:
        placeholder.empty()
        st.altair_chart(alt.Chart(sorted_df.reset_index()).mark_bar().encode(
            x=alt.X('index', sort=None, title=None),
            y=alt.Y('contribution', title="contribution"),
        ).properties(height=300), use_container_width=True)


def main_loop(logl, lglk_threshold, decomp_loglks, agg_window, slice_df, col21, col22, col3):

    # Impute extreme values for visualization purposes
    logl[np.where(logl < -5e4)[0]] = -5e4
    logl_df = pd.DataFrame(np.stack([np.repeat(lglk_threshold, len(
        logl)), logl.copy()], axis=1), columns=['anomaly threshold', 'logl'])

    # Remove suffixes _t_n and sum same-named cols
    decomp_loglks.columns = [re.split(r'_t_(\d+)', x)[0]
                             for x in decomp_loglks.columns]
    decomp_loglks = decomp_loglks.groupby(decomp_loglks.columns, axis=1).sum()

    logl_df.index = slice_df.index[dbn_size-1:]
    # Fire only when thresh is surpassed
    anomaly_fires_idx = np.nonzero(logl_df['logl'] < lglk_threshold)[0]
    decomp_loglks.index = slice_df.index[dbn_size-1:]

    with open('progress_idx', 'rb') as f:
        st.session_state.progress_idx = pickle.load(f)
    with col21:
        # Initial plot
        chart = st.line_chart(logl_df.iloc[:st.session_state.progress_idx],
                              use_container_width=True, color=('#ffcc00', '#00a6ff'))
    with col22:
        placeholder = st.empty()
    if st.session_state.feature is not None:
        with col3:
            start_slice_decomp_logl = decomp_loglks.iloc[:st.session_state.progress_idx,
                                                         decomp_loglks.columns == st.session_state.feature]
            start_slice_feature = slice_df.iloc[:st.session_state.progress_idx +
                                                dbn_size-1, slice_df.columns == st.session_state.feature]
            # Format to avoid encoding exceptions in streamlit
            start_slice_decomp_logl.columns, start_slice_feature.columns = [
                st.session_state.feature.replace('.', '')], [st.session_state.feature.replace('.', '')]
            # Initial plots
            st.write('Feature log-likelihood')
            chart2 = st.line_chart(
                start_slice_decomp_logl, use_container_width=True, height=200)
            st.write('Feature value')
            chart3 = st.line_chart(start_slice_feature,
                                   use_container_width=True, height=200)
    while True:
        plot_ranking(decomp_loglks, anomaly_fires_idx, agg_window, placeholder)
        if st.session_state.play and st.session_state.progress_idx < len(logl):
            plot_logl(chart, logl_df, col21)
            if st.session_state.feature is not None:
                plot_feature(chart2, chart3, decomp_loglks, slice_df, col3)
            # Next time-step
            st.session_state.progress_idx += 1
            time.sleep(st.session_state.speed)
        else:
            break

### CALLBACKS ###
def start_click():
    if st.session_state.play:
        with open('progress_idx', 'wb') as f:
            pickle.dump(2, f)

    st.session_state.play = True


def feature_select():
    with open('progress_idx', 'wb') as f:
        pickle.dump(st.session_state.progress_idx, f)


def stop_click():
    st.session_state.play = False
    feature_select()


def anomaly_select():
    st.session_state.play = False
    with open('progress_idx', 'wb') as f:
        pickle.dump(2, f)


def change_speed():
    match st.session_state.speed_input:
        case 'Low':
            st.session_state.speed = 0.5
        case 'Medium':
            st.session_state.speed = 0.2
        case 'High':
            st.session_state.speed = 0.05
    with open('progress_idx', 'wb') as f:
        pickle.dump(st.session_state.progress_idx, f)

# Change in between fan IDs
def toggle_mode():
    if st.session_state.fan == 1:
        st.session_state.fan = 2
    else:
        st.session_state.fan = 1


### GUI DEFINITION ###
button_text = f'Switch to IDF2' if st.session_state.fan == 1 else 'Switch to IDF1'
col01, col02 = st.columns([7, 1])

# Title
with col01:
    st.title(f'ArcelorMittal - DEMO POC2 - DGBN - IDF{st.session_state.fan}')
# Fan selection
with col02:
    st.button(button_text, type='primary', on_click=toggle_mode)
col11, col12, col13, _ = st.columns([1.5, 1, 1, 5])

with col11:
    # Date selection
    custom_anomaly = None
    anomaly = st.selectbox(
        "Please select anomaly date",
        [x.strftime('%Y/%m/%d') for x in anomalies], key='anomaly', on_change=anomaly_select)
    custom_anomaly = st.date_input("Or enter any valid date", value=None, min_value=dfCV.index.date.min(
    ), max_value=dfCV.index.date.max(), key='custom_anomaly', on_change=anomaly_select)
    if custom_anomaly is not None:
        anomaly = custom_anomaly
    logl, decomp_loglks, slice_df = eval_anomaly(
        pd.to_datetime(anomaly, format='%Y/%m/%d'))


with col12:
    col121, col122 = st.columns([1, 1])

    # Play/Stop triggers
    with col121:
        st.button('Start', key=None, help=None, on_click=start_click, args=None,
                  kwargs=None, type="secondary", disabled=False, use_container_width=False)
    with col122:
        st.button('Stop', key=None, help=None, on_click=stop_click, args=None,
                  kwargs=None, type="secondary", disabled=False, use_container_width=False)
    if 'speed' not in st.session_state:
        st.session_state.speed = 0.5

    # Speed selection
    speed = st.select_slider(
        "Speed",
        options=["Low", "Medium", "High"],
        on_change=change_speed, key='speed_input')

with col13:
    # Likelihood threshold selection for lineplot
    lglk_threshold = st.number_input("Anomaly threshold (nits)", value=-700, max_value=0,
                                     min_value=int(-5e4 - 1), step=100, on_change=feature_select, key='lglk_threshold')


st.divider()

col21, col22 = st.columns([3, 1])
with col21:
    st.write("Log-likelihood of the data")
    _, col212 = st.columns([0.5, 20])
    with col212:
        agg_window = None
        # Show agg_window selector only at stop mode
        if st.session_state.play:
            st.empty()
        else:
            if st.session_state.progress_idx > dbn_size:
                min_value_set = slice_df.index[st.session_state.progress_idx -
                                               1].to_pydatetime().time()
            else:
                min_value_set = slice_df.index[dbn_size -
                                               1].to_pydatetime().time()
            # Aggregation window selection with step 2 minutes
            agg_window = st.slider(
                "Aggregation window",
                step=datetime.timedelta(minutes=2),
                min_value=slice_df.index[dbn_size-1].to_pydatetime().time(), max_value=slice_df.index[st.session_state.progress_idx + dbn_size-2].to_pydatetime().time(),
                value=(min_value_set, slice_df.index[st.session_state.progress_idx + dbn_size-2].to_pydatetime().time()))

with col22:
    col221, col222 = st.columns([1, 1])
    with col221:
        st.write("Feature Importance")
    with col222:
        # k number of features selection
        st.selectbox('Top',
                     options=np.arange(1, 31), key='k', index=9)
st.divider()
col3, _ = st.columns([3, 1])
with col3:
    # Feature detail inspection selection
    anomaly = st.selectbox(
        "Feature detail",
        sorted(slice_df.columns), key='feature', index=None, on_change=feature_select)

st.divider()


logl_df_history = get_history_flags(
    st.session_state.lglk_threshold_history, st.session_state.fan)
col4, _ = st.columns([3, 1])
# 720 time-steps as max_value means a whole day (1440 minutes in a day)
fig = calplot(
    logl_df_history.reset_index(),
    x="index",
    y="logl",
    colorscale="PuBu",
    dark_theme=True,
    gap=0.1,
    years_title=True,
    date_fmt="%Y/%m/%d",
    space_between_plots=0.13,
    cmap_min=0,
    cmap_max=720,

)


with col4:
    st.write(f'Anomaly history for IDF{st.session_state.fan}')
    col41, _ = st.columns([2, 7])
    with col41:
        # Likelihood threshold selection for calendar plot
        lglk_threshold_history = st.number_input("Visualize for anomaly threshold (nits)", max_value=0, min_value=int(
            -5e4 - 1), step=100, on_change=feature_select, key='lglk_threshold_history')
    col42, _ = st.columns([3, 1])
    with col42:
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')

main_loop(logl, lglk_threshold, decomp_loglks,
          agg_window, slice_df, col21, col22, col3)
