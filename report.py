import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from IPython.display import Image
import pickle
import plotly.io as pio

import pickle
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

with open(r'datasets/df.pkl', 'rb') as f:
    db = pickle.load(f)

# Calculate statistics
mean_price = db['P [$/kWh]'].mean()
median_price = db['P [$/kWh]'].median()
std_price = db['P [$/kWh]'].std()

# Generate the distribution plot
fig_distribution = ff.create_distplot([db['P [$/kWh]'].values], ['P [$/kWh]'], bin_size=0.005)
fig_distribution.update_layout(
    title='Distribution of Hydro-Quebec Wholesale Electricity Rate',
    xaxis_title='Price [$/kWh]',
    yaxis_title='Density',
    legend=dict(font=dict(size=18))  # Increase legend font size
)

# Manually calculate the max y-value for the distribution plot
hist_data = np.histogram(db['P [$/kWh]'], bins=int((db['P [$/kWh]'].max() - db['P [$/kWh]'].min()) / 0.005))
max_y_value = max(hist_data[0]) / len(db['P [$/kWh]']) / 0.005

# Add lines for mean, median, and std deviation
fig_distribution.add_trace(go.Scatter(
    x=[mean_price, mean_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean",
    line=dict(color="Red", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[median_price, median_price],
    y=[0, max_y_value],
    mode="lines",
    name="Median",
    line=dict(color="Green", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price - std_price, mean_price - std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean - Std Dev",
    line=dict(color="Blue", width=2, dash="dash")
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price + std_price, mean_price + std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean + Std Dev",
    line=dict(color="Orange", width=2, dash="dash")
))

# Show the distribution plot
fig_distribution.show()

print(f"Mean Price: {mean_price:.6f} $/kWh")
print(f"Median Price: {median_price:.6f} $/kWh")
print(f"Standard Deviation: {std_price:.6f} $/kWh")
print(f"Minimum Price: {db['P [$/kWh]'].min():.6f} $/kWh")
print(f"Maximum Price: {db['P [$/kWh]'].max():.6f} $/kWh")

# Save the figure with DPI 300
pio.write_image(fig_distribution, 'figures/rate_distribution_plot.png', scale=2, width=1200, height=400)

# Filter the DataFrame to include only values within 2 standard deviations from the mean
db = db[(db['P [$/kWh]'] >= mean_price - 2 * std_price) & (db['P [$/kWh]'] <= mean_price + 2 * std_price)]

# Generate the distribution plot
fig_distribution = ff.create_distplot([db['P [$/kWh]'].values], ['P [$/kWh]'], bin_size=0.005)
fig_distribution.update_layout(
    xaxis_title='Price [$/kWh]',
    yaxis_title='Density',
    legend=dict(font=dict(size=18))  # Increase legend font size
)

# Manually calculate the max y-value for the distribution plot
hist_data = np.histogram(db['P [$/kWh]'], bins=int((db['P [$/kWh]'].max() - db['P [$/kWh]'].min()) / 0.005))
max_y_value = max(hist_data[0]) / len(db['P [$/kWh]']) / 0.005

# Add lines for mean, median, and std deviation
fig_distribution.add_trace(go.Scatter(
    x=[mean_price, mean_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean",
    line=dict(color="Red", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[median_price, median_price],
    y=[0, max_y_value],
    mode="lines",
    name="Median",
    line=dict(color="Green", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price - std_price, mean_price - std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean - Std Dev",
    line=dict(color="Blue", width=2, dash="dash")
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price + std_price, mean_price + std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean + Std Dev",
    line=dict(color="Orange", width=2, dash="dash")
))

pio.write_image(fig_distribution, 'figures/rate_distribution_plot_zoomed.png', scale=2, width=1200, height=400)

# Show the distribution plot
fig_distribution.show()

import pickle
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

with open(r'datasets/df.pkl', 'rb') as f:
    db = pickle.load(f)

# Calculate statistics
mean_price = db['Total Emission [gCO₂eq/kWh]'].mean()
median_price = db['Total Emission [gCO₂eq/kWh]'].median()
std_price = db['Total Emission [gCO₂eq/kWh]'].std()

# Generate the distribution plot
fig_distribution = ff.create_distplot([db['Total Emission [gCO₂eq/kWh]'].values], ['Total Emission [gCO₂eq/kWh]'], bin_size=0.005)
fig_distribution.update_layout(
    title='Distribution of Hydro-Quebec Electricity Total Emission',
    xaxis_title='Total Emission [gCO₂eq/kWh]',
    yaxis_title='Density',
    legend=dict(font=dict(size=18))  # Increase legend font size
)

# Manually calculate the max y-value for the distribution plot
hist_data = np.histogram(db['Total Emission [gCO₂eq/kWh]'], bins=int((db['Total Emission [gCO₂eq/kWh]'].max() - db['Total Emission [gCO₂eq/kWh]'].min()) / 0.005))
max_y_value = max(hist_data[0]) / len(db['Total Emission [gCO₂eq/kWh]']) / 0.005

# Add lines for mean, median, and std deviation
fig_distribution.add_trace(go.Scatter(
    x=[mean_price, mean_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean",
    line=dict(color="Red", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[median_price, median_price],
    y=[0, max_y_value],
    mode="lines",
    name="Median",
    line=dict(color="Green", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price - std_price, mean_price - std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean - Std Dev",
    line=dict(color="Blue", width=2, dash="dash")
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price + std_price, mean_price + std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean + Std Dev",
    line=dict(color="Orange", width=2, dash="dash")
))

# Show the distribution plot
fig_distribution.show()

print(f"Mean Price: {mean_price:.6f} $/kWh")
print(f"Median Price: {median_price:.6f} $/kWh")
print(f"Standard Deviation: {std_price:.6f} $/kWh")
print(f"Minimum Price: {db['Total Emission [gCO₂eq/kWh]'].min():.6f} $/kWh")
print(f"Maximum Price: {db['Total Emission [gCO₂eq/kWh]'].max():.6f} $/kWh")

# Save the figure with DPI 300
pio.write_image(fig_distribution, 'figures/co2_distribution_plot.png', scale=2, width=1200, height=400)

# Filter the DataFrame to include only values within 2 standard deviations from the mean
db = db[(db['Total Emission [gCO₂eq/kWh]'] >= mean_price - 2 * std_price) & (db['Total Emission [gCO₂eq/kWh]'] <= mean_price + 2 * std_price)]

# Generate the distribution plot
fig_distribution = ff.create_distplot([db['Total Emission [gCO₂eq/kWh]'].values], ['Total Emission [gCO₂eq/kWh]'], bin_size=0.005)
fig_distribution.update_layout(
    xaxis_title='Total Emission [gCO₂eq/kWh]',
    yaxis_title='Density',
    legend=dict(font=dict(size=18))  # Increase legend font size
)

# Manually calculate the max y-value for the distribution plot
hist_data = np.histogram(db['Total Emission [gCO₂eq/kWh]'], bins=int((db['Total Emission [gCO₂eq/kWh]'].max() - db['Total Emission [gCO₂eq/kWh]'].min()) / 0.005))
max_y_value = max(hist_data[0]) / len(db['Total Emission [gCO₂eq/kWh]']) / 0.005

# Add lines for mean, median, and std deviation
fig_distribution.add_trace(go.Scatter(
    x=[mean_price, mean_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean",
    line=dict(color="Red", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[median_price, median_price],
    y=[0, max_y_value],
    mode="lines",
    name="Median",
    line=dict(color="Green", width=2)
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price - std_price, mean_price - std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean - Std Dev",
    line=dict(color="Blue", width=2, dash="dash")
))

fig_distribution.add_trace(go.Scatter(
    x=[mean_price + std_price, mean_price + std_price],
    y=[0, max_y_value],
    mode="lines",
    name="Mean + Std Dev",
    line=dict(color="Orange", width=2, dash="dash")
))

pio.write_image(fig_distribution, 'figures/co2_distribution_plot_zoomed.png', scale=2, width=1200, height=400)

# Show the distribution plot
fig_distribution.show()

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
def hvac_alg_comparision(alg_name):
    with open(r'res/agg_res.pkl', 'rb') as f:
        res = pickle.load(f)
    # temperatures

    # Assuming `res` is your dictionary and `res['db']` is a dataframe
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['db'].index, y=res['db']['Des Temp [°C]'], name='desired indoor temperature'))
    fig.add_trace(go.Scatter(x=res['db'].index, y=res['agg_main_Tin'], name='indoor temperature'))
    fig.add_trace(go.Scatter(x=res['db'].index, y=res['db']['Temp [°C]'], name='outdoor temperature'))

    fig.update_layout(xaxis_title='Time', yaxis_title='Temperature [°C]')

    # Identify each unique day in the data
    unique_days = res['db'].index.normalize().unique()

    for day in unique_days:
        # Green shade from 6:00 to 9:00 AM
        start_morning = pd.Timestamp(day) + pd.Timedelta(hours=6)
        end_morning = pd.Timestamp(day) + pd.Timedelta(hours=9)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_morning,
            x1=end_morning,
            y0=0,
            y1=1,
            fillcolor="green",
            opacity=0.15,
            layer="below",
            line_width=0
        )

        # Purple shade from 4:00 to 8:00 PM
        start_evening = pd.Timestamp(day) + pd.Timedelta(hours=16)
        end_evening = pd.Timestamp(day) + pd.Timedelta(hours=20)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start_evening,
            x1=end_evening,
            y0=0,
            y1=1,
            fillcolor="purple",
            opacity=0.15,
            layer="below",
            line_width=0
        )

    # Save the figure as an image and then show it
    pio.write_image(fig, f'figures/temperature_{alg_name}.png', scale=2, width=1200, height=400)
    fig.show()

    print(f'Algorithm name: {alg_name}')
    print(f'total cost: {res["agg_total_cost"]} (change %: {100*(56820.615629782995-res["agg_total_cost"])/56820.615629782995})')
    print(f'Average indoor temperature: {res["agg_main_Tin"].mean()} (change %: {100*(res["agg_main_Tin"].mean()-res["agg_ref_Tin"].mean())/res["agg_ref_Tin"].mean()})')

    main_load_peak_hours = res['agg_load_total'] * res['db']['peak']
    ref_load_peak_hours = res['agg_ref_total_load'] * res['db']['peak']
    diff_load_peak_hours = ref_load_peak_hours - main_load_peak_hours
    shaved_peak = ref_load_peak_hours.max()-main_load_peak_hours.max()
    print(f'total saved consumption during peak hours: {diff_load_peak_hours.sum()} (change %: {100*(diff_load_peak_hours.sum())/ref_load_peak_hours.sum()})')
    print(f'peak load shaved: {shaved_peak} (change %: {100*shaved_peak/ref_load_peak_hours.max()})')
    print(f'cost components: {res["agg_cost_components"].sum(axis=0)}')
    print(f'cost components (change %): {[None, 100*(49870.05230516-res["agg_cost_components"].sum(axis=0)[1])/49870.05230516, 100*(13227.94980496-res["agg_cost_components"].sum(axis=0)[2])/13227.94980496, 100*(2637.83655794-res["agg_cost_components"].sum(axis=0)[3])/2637.83655794]}')

# Status Quo (OR Model)
hvac_alg_comparision('OR_model')
# turn hvac off
hvac_alg_comparision('HVAC_OFF')
# hueristic 0
hvac_alg_comparision('Heuristic_0')
# hueristic -1
hvac_alg_comparision('Heuristic_1')
# hueristic -2
hvac_alg_comparision('Heuristic_2')
# CTRL Only TD3
hvac_alg_comparision('Controllable_Only')
# heuristic 2 and peak
hvac_alg_comparision('Heuristic-2_peak')
# heuristic 2 and peak and TD3
hvac_alg_comparision('Heuristic-2_peak_TD3')
# d3qn
hvac_alg_comparision('DDPG')
# ppod
hvac_alg_comparision('PPOD')
# sacd
hvac_alg_comparision('SACD')

import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

with open(r'res/agg_res.pkl', 'rb') as f:
    res = pickle.load(f)

fig = go.Figure()

# Add reference controllable load as an area chart
fig.add_trace(go.Scatter(
    x=res['db'].index,
    y=res['agg_ref_ctrl_load'],
    name='Reference Controllable Load',
    fill='tozeroy',  # fills the area below the line
    mode='none'
))

# Add new controllable load as a line chart
fig.add_trace(go.Scatter(
    x=res['db'].index,
    y=res['agg_load_ctrl'],
    name='New Controllable Load',
    mode='lines'
))

# Update layout to add axis labels
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Load [kWh]'
)

# Identify each unique day in the data
unique_days = res['db'].index.normalize().unique()

for day in unique_days:
    # Green shade from 6:00 to 9:00 AM
    start_morning = pd.Timestamp(day) + pd.Timedelta(hours=6)
    end_morning = pd.Timestamp(day) + pd.Timedelta(hours=9)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_morning,
        x1=end_morning,
        y0=0,
        y1=1,
        fillcolor="green",
        opacity=0.15,
        layer="below",
        line_width=0
    )

    # Purple shade from 4:00 to 8:00 PM
    start_evening = pd.Timestamp(day) + pd.Timedelta(hours=16)
    end_evening = pd.Timestamp(day) + pd.Timedelta(hours=20)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_evening,
        x1=end_evening,
        y0=0,
        y1=1,
        fillcolor="purple",
        opacity=0.15,
        layer="below",
        line_width=0
    )

pio.write_image(fig, 'figures/controllable_load.png', scale=2, width=1200, height=400)
fig.show()

import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd

# Assuming res is already loaded and available
# res = ... (your data loading step)

trace_nonreducible = go.Scatter(
    x=res['db'].index,
    y=res['agg_total_nonreducible'],
    mode='lines',
    name='Non-Reducible',
    stackgroup='one'
)

trace_ctrl = go.Scatter(
    x=res['db'].index,
    y=res['agg_load_ctrl'],
    mode='lines',
    name='Controllable',
    stackgroup='one'
)

trace_hvac = go.Scatter(
    x=res['db'].index,
    y=res['agg_load_hvac'],
    mode='lines',
    name='HVAC',
    stackgroup='one'
)

# Combine traces into a figure
fig = go.Figure(data=[trace_nonreducible, trace_ctrl, trace_hvac])

# Update layout
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Load [kWh]',
    template='plotly_white'
)

# Identify each unique day in the data
unique_days = res['db'].index.normalize().unique()

for day in unique_days:
    # Green shade from 6:00 to 9:00 AM
    start_morning = pd.Timestamp(day) + pd.Timedelta(hours=6)
    end_morning = pd.Timestamp(day) + pd.Timedelta(hours=9)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_morning,
        x1=end_morning,
        y0=0,
        y1=1,
        fillcolor="green",
        opacity=0.15,
        layer="below",
        line_width=0
    )

    # Purple shade from 4:00 to 8:00 PM
    start_evening = pd.Timestamp(day) + pd.Timedelta(hours=16)
    end_evening = pd.Timestamp(day) + pd.Timedelta(hours=20)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start_evening,
        x1=end_evening,
        y0=0,
        y1=1,
        fillcolor="purple",
        opacity=0.15,
        layer="below",
        line_width=0
    )

pio.write_image(fig, 'figures/total_load.png', scale=2, width=1200, height=400)
fig.show()

import pickle
import numpy as np
import plotly.graph_objs as go
import itertools
import plotly.io as pio

# Define parameter values
C_values = [100, 150]
R_values = [1, 2]
h_values = [50, 80]
alpha_values = [0.2, 0.3]
beta_values = [0.2, 0.3]
gamma = 0.55132

new_time_steps = range(0, 200000, 50)
d3qn_ = np.zeros_like(new_time_steps, dtype=np.float64)
ppod_ = np.zeros_like(new_time_steps, dtype=np.float64)
sacd_ = np.zeros_like(new_time_steps, dtype=np.float64)
cc = 0
# Iterate over all combinations of parameters
for C, R, h, alpha in itertools.product(C_values, R_values, h_values, alpha_values):
    # Load the data for each algorithm
    with open(f'res/track/track_HVAC_D3QN_C_{C}_R_{R}_h_{h}_alpha_{alpha}_gamma_{gamma}.pkl', 'rb') as f:
        d3qn = pickle.load(f)
    best_results = []

    # Iterate over the new time steps
    current_best = float('-inf')
    for time_step in new_time_steps:
        # Find the best result up to the current time step
        for i, time in enumerate(d3qn[0]):
            if time <= time_step:
                current_best = max(current_best, d3qn[1][i])
            else:
                break
        best_results.append(current_best)
    d3qn_ += best_results

    with open(f'res/track/track_HVAC_PPOD_C_{C}_R_{R}_h_{h}_alpha_{alpha}_gamma_{gamma}.pkl', 'rb') as f:
        ppod = pickle.load(f)
    best_results = []

    # Iterate over the new time steps
    current_best = float('-inf')
    for time_step in new_time_steps:
        # Find the best result up to the current time step
        for i, time in enumerate(ppod[0]):
            if time <= time_step:
                current_best = max(current_best, ppod[1][i])
            else:
                break
        best_results.append(current_best)
    ppod_ += best_results
    with open(f'res/track/track_HVAC_SACD_C_{C}_R_{R}_h_{h}_alpha_{alpha}_gamma_{gamma}.pkl', 'rb') as f:
        sacd = pickle.load(f)
    best_results = []

    # Iterate over the new time steps
    current_best = float('-inf')
    for time_step in new_time_steps:
        # Find the best result up to the current time step
        for i, time in enumerate(sacd[0]):
            if time <= time_step:
                current_best = max(current_best, sacd[1][i])
            else:
                break
        best_results.append(current_best)
    sacd_ += best_results

    cc += 1

fig = go.Figure()

# Add traces for each algorithm
fig.add_trace(go.Scatter(x=list(new_time_steps), y=d3qn_/cc, mode='lines', name='D3QN'))
fig.add_trace(go.Scatter(x=list(new_time_steps), y=ppod_/cc, mode='lines', name='PPOD'))
fig.add_trace(go.Scatter(x=list(new_time_steps), y=sacd_/cc, mode='lines', name='SACD'))

# Update layout
fig.update_layout(xaxis_title='Elapsed Time',yaxis_title='Average Best Reward Obtained', title=f'HVAC Algorithms Performance')

# Show plot
fig.show()
pio.write_image(fig, f'figures/hvac_algs_performance.png', scale=2, width=1200, height=400)

import pickle
import numpy as np
import plotly.graph_objs as go
import itertools
import plotly.io as pio

# Define parameter values
beta_values = [0.2, 0.3]
gamma = 0.55132

new_time_steps = range(0, 100000, 50)
ddpg_ = np.zeros_like(new_time_steps, dtype=np.float64)
ppo_ = np.zeros_like(new_time_steps, dtype=np.float64)
td3_ = np.zeros_like(new_time_steps, dtype=np.float64)

# Iterate over all combinations of parameters
for beta in beta_values:
    # Load the data for each algorithm
    with open(f'res/track/track_CTRL_DDPG_beta_{beta}_gamma_{gamma}.pkl', 'rb') as f:
        ddpg = pickle.load(f)

    best_results = np.full(len(new_time_steps), float('-inf'), dtype=np.float64)

    # Iterate over the new time steps
    for idx, time_step in enumerate(new_time_steps):
        # Find the best result up to the current time step
        for i, time in enumerate(ddpg[0]):
            if time <= time_step:
                best_results[idx] = max(best_results[idx], ddpg[1][i].item())
            else:
                break

    # Add to ddpg_
    ddpg_ += best_results

    with open(f'res/track/track_CTRL_PPO_beta_{beta}_gamma_{gamma}.pkl', 'rb') as f:
        ppo = pickle.load(f)

    best_results = np.full(len(new_time_steps), float('-inf'), dtype=np.float64)

    # Iterate over the new time steps
    for idx, time_step in enumerate(new_time_steps):
        # Find the best result up to the current time step
        for i, time in enumerate(ppo[0]):
            if time <= time_step:
                best_results[idx] = max(best_results[idx], ppo[1][i].item())
            else:
                break

    ppo_ += best_results

    with open(f'res/track/track_CTRL_TD3_{beta}_{gamma}.pkl', 'rb') as f:
        td3 = pickle.load(f)

    best_results = np.full(len(new_time_steps), float('-inf'), dtype=np.float64)

    # Iterate over the new time steps
    for idx, time_step in enumerate(new_time_steps):
        # Find the best result up to the current time step
        for i, time in enumerate(td3[0]):
            if time <= time_step:
                best_results[idx] = max(best_results[idx], td3[1][i].item())
            else:
                break

    # Add to ddpg_
    td3_ += best_results

    cc += 1

import plotly.graph_objects as go
import plotly.io as pio

fig = go.Figure()

# Add traces for each algorithm
fig.add_trace(go.Scatter(x=list(new_time_steps), y=ddpg_ / cc, mode='lines', name='DDPG'))
fig.add_trace(go.Scatter(x=list(new_time_steps), y=ppo_ / cc, mode='lines', name='PPO'))
fig.add_trace(go.Scatter(x=list(new_time_steps), y=td3_ / cc, mode='lines', name='TD3'))

# Update layout
fig.update_layout(
    xaxis_title='Elapsed Time',
    yaxis_title='Average Best Reward Obtained',
    title='CTRL Algorithms Performance',
    yaxis=dict(range=[-10, -5])
)

# Show plot
fig.show()

# Save plot as image
pio.write_image(fig, 'figures/ctrl_algs_performance.png', scale=2, width=1200, height=400)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

algorithms = ["Heuristic_0", "Heuristic_1", "Heuristic_2", "TD3", "Heuristic_2_peak", "Heuristic_2_peak_TD3", "D3QN_TD3", "PPOD_TD3", "SACD_TD3"]
# total_cost_change = [-6.0366426412441285, 11.162125606943738, 27.85371431635701, 10.63180851576139, 37.7504987896538, 48.382307305415175, 49.90187926606712, 66.79415288678616, 63.766719557467674]
total_saved_consumption_peak_hours = [-5319.999999999993, 8680.0, 23440.00000000001, 10048.680767867318, 36640.00000000001, 46688.680767867336, 48308.68076786732, 65068.68076786735, 61568.680767867336]
peak_load_shaved_change = [4.128990444224641, 7.935176024146247, 3.8396871516776523, 0.0, 11.483524741067322, 11.483524741067322, 24.40339023382219, 27.07334026325346, 20.60773637643016]
total_cost_change = [-0.9966604989394381, 3.121975645236912, 5.822554852250698, 1.0046487699526807, 2.505914788191865, 3.5105635581463694, 3.450992220947652, 4.169200140538759, 4.589118979241064]
dissatisfaction_change = [-42.935665189834275, -38.18746836546275, -266.06977577530796, -32.82375207504142, -84.3846402919376, -117.2083923669498, -220.2205607964661, -138.6004530303827, -176.518830309146]
co2_emission_change = [-1.7164074576099708, 1.5595132158382925, 4.889953628035079, 0.7869537368410208, 1.3331944604667283, 2.1201481972100535, 1.1836251454321012, 1.5564150264832823, 2.5659645177071626]
dissatisfaction_change = -np.array(dissatisfaction_change)

# Define colors for algorithms
colors = px.colors.qualitative.Plotly

# Create subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=(
    'Total Cost decrease (%)',
    # 'Total Saved Consumption During Peak Hours (kWh)',
    'Peak Load Shaved (Ratio %)',
    'Dissatisfaction increase (%)',
    'CO2 Emission decrease (%)'
))

# Total Cost Change (%)
fig.add_trace(go.Bar(
    x=algorithms,
    y=total_cost_change,
    marker_color=colors[:len(algorithms)],
    showlegend=False
), row=1, col=1)

# # Total Saved Consumption During Peak Hours
# fig.add_trace(go.Bar(
#     x=algorithms,
#     y=total_saved_consumption_peak_hours,
#     marker_color=colors[:len(algorithms)],
#     showlegend=False
# ), row=1, col=2)

# Peak Load Shaved Change (%)
fig.add_trace(go.Bar(
    x=algorithms,
    y=peak_load_shaved_change,
    marker_color=colors[:len(algorithms)],
    showlegend=False
), row=1, col=2)

# Dissatisfaction Change (%)
fig.add_trace(go.Bar(
    x=algorithms,
    y=dissatisfaction_change,
    marker_color=colors[:len(algorithms)],
    showlegend=False
), row=2, col=1)

# CO2 Emission Change (%)
fig.add_trace(go.Bar(
    x=algorithms,
    y=co2_emission_change,
    marker_color=colors[:len(algorithms)],
    showlegend=False
), row=2, col=2)

# Update layout
fig.update_layout(height=800, width=1200)

# Hide unused subplot
# fig.update_xaxes(visible=False, row=2, col=2)
# fig.update_yaxes(visible=False, row=2, col=2)

pio.write_image(fig, f'figures/algs_comparison.png', scale=2, width=1200, height=800)

# Show the plot
fig.show()

import matplotlib.pyplot as plt
import numpy as np

# Data for the star plot
categories = ['Avg. Indoor Temp (°C)', 'Change in Saved Consumption (%)',
              'Peak Load Shaved (%)', 'GHG Emission Reduction (%)']

# Values for each parameter
alpha_values = [[19.18, 13.97, 25.77, 1.42],
                [19.15, 13.29, 18.73, 1.58],
                [19.14, 12.77, 27.83, 1.55]]

beta_values = [[19.15, 14.60, 18.73, 2.04],
               [19.15, 13.29, 18.73, 1.58],
               [19.15, 13.17, 18.73, 1.53]]

gamma_values = [[19.08, 10.68, 18.44, 1.75],
                [19.17, 11.26, 27.04, 1.41],
                [19.15, 13.29, 18.73, 1.58],
                [19.11, 13.40, 12.81, 1.91],
                [19.18, 14.43, 28.72, 1.89]]

parameter_groups = [('α', alpha_values), ('β', beta_values), ('γ', gamma_values)]

# Create a 1×3 layout
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

for idx, (parameter, values) in enumerate(parameter_groups):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax = axs[idx]

    for value_set in values:
        stats = value_set + value_set[:1]
        ax.plot(angles, stats, label=f'{parameter} ({values.index(value_set)+1})', linewidth=2)
        # Removed ax.fill(angles, stats, alpha=0.25) to show only lines

    ax.set_yticks([10, 15, 20, 25])  # Adjust as necessary
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)  # Adjust font size for readability
    ax.set_title(f'Parameter {parameter}', va='bottom', fontsize=12)
    ax.tick_params(axis='x', pad=10)  # Add padding to prevent collision

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout(pad=3.0)  # Adjust spacing between plots

# Save the plot
file_path = "figures/Star_Diagrams_Parameters_LinesOnly.png"
plt.savefig(file_path, bbox_inches='tight', dpi=300)
plt.show()

hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_alpha_0.1_0.2')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_alpha_0.3_0.4')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_beta_0.1_0.2')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_beta_0.3_0.4')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_gamma_0.35132')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_gamma_0.45132')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_gamma_0.55132')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_gamma_0.65132')
hvac_alg_comparision('HVAC_PPOD_CTRL_TD3_gamma_0.75132')
import matplotlib.pyplot as plt

# Data for the different alpha ranges
alpha_ranges = ['0.1-0.2', '0.2-0.3', '0.3-0.4']
avg_temp = [19.18, 19.15, 19.14]
saved_consumption = [13.97, 13.29, 12.77]
peak_load_shaved = [25.77, 18.73, 27.83]
ghg_reduction = [1.42, 1.58, 1.55]

# Create subplots
fig, ax1 = plt.subplots()

# Plot average indoor temperature
ax1.plot(alpha_ranges, avg_temp, 'bo-', label='Avg Indoor Temp (°C)')
ax1.set_xlabel('Alpha Range')
ax1.set_ylabel('Avg Indoor Temperature (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a twin y-axis to plot other metrics
ax2 = ax1.twinx()
ax2.plot(alpha_ranges, saved_consumption, 'r^-', label='Saved Consumption (%)')
ax2.plot(alpha_ranges, peak_load_shaved, 'gs-', label='Peak Load Shaved (%)')
ax2.plot(alpha_ranges, ghg_reduction, 'md-', label='GHG Emission Reduction (%)')
ax2.set_ylabel('Percentage (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add a title
plt.title('Changes in Metrics with Increasing Alpha')

# Show plot
plt.show()

# Data for the different beta ranges
beta_ranges = ['0.1-0.2', '0.2-0.3', '0.3-0.4']
avg_temp_beta = [19.15, 19.15, 19.15]
saved_consumption_beta = [14.60, 13.29, 13.17]
peak_load_shaved_beta = [18.73, 18.73, 18.73]
ghg_reduction_beta = [2.04, 1.58, 1.53]

# Create subplots for beta data
fig, ax1 = plt.subplots()

# Plot average indoor temperature for beta
ax1.plot(beta_ranges, avg_temp_beta, 'bo-', label='Avg Indoor Temp (°C)')
ax1.set_xlabel('Beta Range')
ax1.set_ylabel('Avg Indoor Temperature (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a twin y-axis to plot other metrics
ax2 = ax1.twinx()
ax2.plot(beta_ranges, saved_consumption_beta, 'r^-', label='Saved Consumption (%)')
ax2.plot(beta_ranges, peak_load_shaved_beta, 'gs-', label='Peak Load Shaved (%)')
ax2.plot(beta_ranges, ghg_reduction_beta, 'md-', label='GHG Emission Reduction (%)')
ax2.set_ylabel('Percentage (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add a title
plt.title('Changes in Metrics with Increasing Beta')

# Show plot
plt.show()

# Data for the different gamma ranges
gamma_ranges = ['0.35132', '0.45132', '0.55132', '0.65132', '0.75132']
avg_temp_gamma = [19.08, 19.17, 19.15, 19.11, 19.18]
saved_consumption_gamma = [10.68, 11.26, 13.29, 13.40, 14.43]
peak_load_shaved_gamma = [18.44, 27.04, 18.73, 12.81, 28.72]
ghg_reduction_gamma = [1.75, 1.41, 1.58, 1.91, 1.89]

# Create subplots for gamma data
fig, ax1 = plt.subplots()

# Plot average indoor temperature for gamma
ax1.plot(gamma_ranges, avg_temp_gamma, 'bo-', label='Avg Indoor Temp (°C)')
ax1.set_xlabel('Gamma Range')
ax1.set_ylabel('Avg Indoor Temperature (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a twin y-axis to plot other metrics
ax2 = ax1.twinx()
ax2.plot(gamma_ranges, saved_consumption_gamma, 'r^-', label='Saved Consumption (%)')
ax2.plot(gamma_ranges, peak_load_shaved_gamma, 'gs-', label='Peak Load Shaved (%)')
ax2.plot(gamma_ranges, ghg_reduction_gamma, 'md-', label='GHG Emission Reduction (%)')
ax2.set_ylabel('Percentage (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add a title
plt.title('Changes in Metrics with Increasing Gamma')

# Show plot
plt.show()
