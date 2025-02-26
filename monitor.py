import time

import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import plotly.express as px


def transform(df, dt=0.1):
    def R_x(x):
        # body frame rotation about x-axis
        return np.array([[1, 0, 0],
                         [0, np.cos(-x), -np.sin(-x)],
                         [0, np.sin(-x), np.cos(-x)]])

    def R_y(y):
        # body frame rotation about y-axis
        return np.array([[np.cos(-y), 0, -np.sin(-y)],
                         [0, 1, 0],
                         [np.sin(-y), 0, np.cos(-y)]])

    def R_z(z):
        # body frame rotation about z axis
        return np.array([[np.cos(-z), -np.sin(-z), 0],
                         [np.sin(-z), np.cos(-z), 0],
                         [0, 0, 1]])

    # roll = x, pitch = y, yaw = z, converted to radians
    roll = df['gx']
    pitch = df['gy']
    yaw = df['gz']

    accel = np.array([df['ax'], df['ay'], df['az']])
    earth_accel = np.empty(accel.shape)

    for i in range(df.shape[0]):
        earth_accel[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ accel[:, i]

    df['eax'] = earth_accel[0, :]
    df['eay'] = earth_accel[1, :]
    df['eaz'] = earth_accel[2, :]

    # calculate position coordinates
    return pd.DataFrame({
        'x': integrate.cumulative_trapezoid(integrate.cumulative_trapezoid(df['eax'], dx=dt), dx=dt),
        'y': integrate.cumulative_trapezoid(integrate.cumulative_trapezoid(df['eay'], dx=dt), dx=dt),
        'z': integrate.cumulative_trapezoid(integrate.cumulative_trapezoid(df['eaz'], dx=dt), dx=dt),
    })


@st.cache_data
def get_data():
    with duckdb.connect("sensor.db") as db:
        df = db.execute("select * from measurements").fetchdf()
        return df


hist_height = 250
model_height = 600
step = 40

if 'sensor' not in st.session_state:
    st.set_page_config(layout="wide")
    st.session_state.sensor = {
        'data': get_data(),
        'first': -step,
        'last': 0
    }

df = st.session_state.sensor['data']
first = st.session_state.sensor['first'] + step
last = first + step
if last >= df.shape[0]:
    last = df.shape[0]
st.session_state.sensor['first'] = first
st.session_state.sensor['last'] = last

df = df.loc[first:last:1].copy().reset_index()
edf = transform(df)


placeholder = st.empty()

with st.container():
    st.plotly_chart(px.line_3d(edf, x='x', y='y', z='z', height=model_height), use_container_width=True)

with st.container():
    ax, ay, az = st.columns(3)
    with ax:
        st.plotly_chart(px.line(df, x='clock', y='ax', title='Acceleration X', height=hist_height))
    with ay:
        st.plotly_chart(px.line(df, x='clock', y='ay', title='Acceleration Y', height=hist_height))
    with az:
        st.plotly_chart(px.line(df, x='clock', y='az', title='Acceleration Z', height=hist_height))

with st.container():
    gx, gy, gz = st.columns(3)
    with gx:
        st.plotly_chart(px.line(df, x='clock', y='gx', title="Gyro X", height=hist_height))
    with gy:
        st.plotly_chart(px.line(df, x='clock', y='gy', title='Gyro Y', height=hist_height))
    with gz:
        st.plotly_chart(px.line(df, x='clock', y='gz', title='Gyro Z', height=hist_height))


time.sleep(2)
st.rerun()