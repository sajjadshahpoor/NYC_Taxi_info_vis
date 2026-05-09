import pandas as pd
import geopandas as gpd
import folium
import json
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, no_update, ctx, State
import numpy as np

def empty_figure(message="No data"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=30))
    return fig

def apply_style_to_figure(fig):
    fig.update_layout(
        font=dict(
            family="Segoe UI",
            size=10
        ),
        title=dict(
            font=dict(
                size=14,
                family="Segoe UI",
            )
        ),
        margin=dict(l=30, r=20, t=35, b=30),
        coloraxis_colorbar=dict(
            title_font=dict(size=10),
            tickfont=dict(size=10)
        )
    )
    return fig

def get_filtered_trips(
    start_date,
    end_date,
    selected_pickup_borough,
    selected_dropoff_borough,
    selected_payment,
    selected_passenger_range,
    selected_hour_weekdays=None,
    selected_month_weekdays=None):

    if start_date is None:
        start_date = min_date
    if end_date is None:
        end_date = max_date

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    filtered = trips_df[
        (trips_df['tpep_pickup_datetime'] >= start) &
        (trips_df['tpep_pickup_datetime'] < end + pd.Timedelta(days=1))
    ].copy()

    if selected_pickup_borough != 'All':
        filtered = filtered[filtered['PULocationBorough'] == selected_pickup_borough]

    if selected_dropoff_borough != 'All':
        filtered = filtered[filtered['DOLocationBorough'] == selected_dropoff_borough]

    if selected_payment != 'All':
        filtered = filtered[filtered['payment_type_name'] == selected_payment]

    filtered = filtered[
        (filtered['passenger_count'] >= selected_passenger_range[0]) &
        (filtered['passenger_count'] <= selected_passenger_range[1])
    ]

    selection_masks = []

    if selected_hour_weekdays:
        selected_pairs = {
            (cell['hour'], cell['weekday'])
            for cell in selected_hour_weekdays
        }
        hour_mask = (
            filtered[['pickup_hour', 'pickup_dayofweek']]
            .apply(tuple, axis=1)
            .isin(selected_pairs)
        )
        selection_masks.append(hour_mask)

    if selected_month_weekdays:
        selected_month_pairs = {
            (cell['month'], cell['weekday'])
            for cell in selected_month_weekdays
        }
        month_mask = (
            filtered[['pickup_month', 'pickup_dayofweek']]
            .apply(tuple, axis=1)
            .isin(selected_month_pairs)
        )
        selection_masks.append(month_mask)

    if selection_masks:
        combined_mask = selection_masks[0]

        for mask in selection_masks[1:]:
            combined_mask = combined_mask | mask

        filtered = filtered[combined_mask]

    return filtered

# ------------------------------
# 1. Load and prepare data
# ------------------------------
print("Loading data...")
trips_df = pd.read_parquet('data/trips_sample.parquet')
zones_gdf = gpd.read_file('data/taxi_zones.geojson')

# Ensure zones have correct CRS
zones_gdf = zones_gdf.to_crs("EPSG:4326")

# Convert datetime columns
trips_df['tpep_pickup_datetime'] = pd.to_datetime(trips_df['tpep_pickup_datetime'])
trips_df['pickup_date'] = trips_df['tpep_pickup_datetime'].dt.date
trips_df['pickup_month'] = trips_df['tpep_pickup_datetime'].dt.month
trips_df['pickup_hour'] = trips_df['tpep_pickup_datetime'].dt.hour
trips_df['pickup_dayofweek'] = trips_df['tpep_pickup_datetime'].dt.dayofweek

# Get tip percentage
trips_df['tip_percentage'] = np.where(
    trips_df['fare_amount'] > 0,
    trips_df['tip_amount'] / trips_df['fare_amount'] * 100,
    0
)

# Get min and max dates for the date picker
min_date = trips_df['tpep_pickup_datetime'].min().date()
max_date = trips_df['tpep_pickup_datetime'].max().date()

# Get max passengers
max_passengers = int(trips_df['passenger_count'].max())

# Create a borough mapping for zones (based on zone name)
borough_keywords = {
    'Manhattan': ['Manhattan', 'Midtown', 'Downtown', 'Upper', 'Lower', 'Chelsea', 'Soho', 'Tribeca', 'Gramercy', 'East Village', 'West Village', 'Harlem', 'Washington Heights', 'Inwood'],
    'Brooklyn': ['Brooklyn', 'Williamsburg', 'Bushwick', 'Bedford-Stuyvesant', 'Crown Heights', 'Flatbush', 'Bay Ridge', 'Coney Island'],
    'Queens': ['Queens', 'Astoria', 'Long Island City', 'Flushing', 'Jamaica', 'Howard Beach'],
    'Bronx': ['Bronx', 'Riverdale', 'Fordham', 'Pelham Bay'],
    'Staten Island': ['Staten Island']
}
def map_zone_to_borough(zone_name):
    for borough, keywords in borough_keywords.items():
        if any(keyword in zone_name for keyword in keywords):
            return borough
    return 'Other'
zones_gdf['borough'] = zones_gdf['zone'].apply(map_zone_to_borough)

# Create a mapping from LocationID to borough for filtering
zone_to_borough = dict(zip(zones_gdf['LocationID'], zones_gdf['borough']))

# Add borough columns to trips
trips_df['PULocationBorough'] = trips_df['PULocationID'].map(zone_to_borough).fillna('Other')
trips_df['DOLocationBorough'] = trips_df['DOLocationID'].map(zone_to_borough).fillna('Other')

# Map payment_type codes to names
payment_map = {1: 'Credit', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
trips_df['payment_type_name'] = trips_df['payment_type'].map(payment_map).fillna('Other')

# Calculate duration
trips_df['tpep_dropoff_datetime'] = pd.to_datetime(trips_df['tpep_dropoff_datetime'])

trips_df['trip_duration_min'] = (
    trips_df['tpep_dropoff_datetime'] - trips_df['tpep_pickup_datetime']
).dt.total_seconds() / 60

trips_df = trips_df[
    (trips_df['trip_duration_min'] > 0)
]

# ------------------------------
# 2. Initialize Dash app
# ------------------------------
app = Dash(__name__)
server = app.server

# ------------------------------
# 3. App layout
# ------------------------------
label_style = {
    'color': 'white',
    'display': 'block',
    'marginBottom': '6px'
}

dropdown_style = {
    'width': '100%',
    'fontSize': '12px'
}

kpi_card_style = {
    'background': 'white',
    'color': 'black',
    'borderRadius': '8px',
    'padding': '8px',
    'fontSize': '12px'
}

nav_button_style = {
    'flex': '1',
}

active_nav_style = {
    **nav_button_style,
    'backgroundColor': '#1f2d3d',
    'color': 'white'
}

inactive_nav_style = {
    **nav_button_style,
    'backgroundColor': '#ecf0f1',
    'color': '#555'
}

heatmap_style = {
    'background': 'white',
    'padding': '6px',
    'boxShadow': '0 2px 6px rgba(0,0,0,0.08)'
}

main_content_style = {
    'display': 'flex',
    'gap': '10px',
    'marginBottom': '8px',
    'marginLeft': '320px'
}

sidebar_style = {
    'width': '280px',
    'height': '100vh',
    'backgroundColor': '#1f2d3d',
    'padding': '20px',
    'boxSizing': 'border-box',
    'position': 'fixed',
    'zIndex': 10000
}

filter_bar =html.Div([
        html.Div([
            html.Label("Date Range:", style=label_style),
            dcc.DatePickerRange(
                id='date-range',
                start_date=min_date,
                end_date=max_date,
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            ),
        ], style={'marginBottom': '18px'}),
        
        html.Div([
            html.Label("Pick-up Borough:", style=label_style),
            dcc.Dropdown(
                id='pickup-borough-dropdown',
                options=[{'label': 'All Pick-up Boroughs', 'value': 'All'}] + 
                        [{'label': b, 'value': b} for b in zones_gdf['borough'].unique() if b != 'Other'],
                value='All',
                clearable=False,
                style=dropdown_style
            ),
        ], style={'marginBottom': '18px'}),

        html.Div([
            html.Label("Drop-off Borough:", style=label_style),
            dcc.Dropdown(
                id='dropoff-borough-dropdown',
                options=[{'label': 'All Drop-off Boroughs', 'value': 'All'}] + 
                        [{'label': b, 'value': b} for b in zones_gdf['borough'].unique() if b != 'Other'],
                value='All',
                clearable=False,
                style=dropdown_style
            ),
        ], style={'marginBottom': '18px'}),

        html.Div([
            html.Label("Payment Method:", style=label_style),
            dcc.Dropdown(
                id='payment-dropdown',
                options=[{'label': 'All Payment Methods', 'value': 'All'}] +
                        [{'label': p, 'value': p} for p in sorted(trips_df['payment_type_name'].unique())],
                value='All',
                clearable=False,
                style=dropdown_style
            ),
        ], style={'marginBottom': '18px'}),

        html.Div([
            html.Label("Passenger Count:", style=label_style),
            dcc.RangeSlider(
                id='passenger-range',
                min=0,
                max=max_passengers,
                step=1,
                value=[0, max_passengers],
                marks={
                    i: {
                        'label': str(i),
                        'style': {'color': 'white'}
                    }
                    for i in range(max_passengers + 1)}
            ),
        ], style={'marginBottom': '18px'}),

        html.Button(
            "↺ Reset",
            id='reset-filters-btn',
            n_clicks=0,
            style={
                'width': '100%',
                'height': '38px',
                'backgroundColor': '#95a5a6',
                'color': 'white',
                'border': 'none',
                'cursor': 'pointer'
            }
        )
    ], style={'padding': '5px', 'borderBottom': '1px solid #ddd', 'marginBottom': '5px', 'flexShrink': '0'})

overview_layout = html.Div([
    # KPI row
    html.Div([
        html.Div(id='total-trips', style=kpi_card_style),
        html.Div(id='total-revenue', style=kpi_card_style),
        html.Div(id='avg-fare', style=kpi_card_style),
        html.Div(id='avg-tip-pct', style=kpi_card_style),
        html.Div(id='avg-distance', style=kpi_card_style),
        html.Div(id='avg-duration', style=kpi_card_style),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(6, 1fr)',
        'width': '100%',
        'marginBottom': '8px',
        'marginTop': '0',
        'paddingTop': '0'
    }),

    # Heatmaps row
    html.Div([
        html.Div([
            html.H4("WHEN IS DEMAND HIGH?", style={'margin': '2px 0'}),
            html.H5(
                "Click on a time slot to filter all visuals",
                style={'marginBottom': '6px'}
            ),
            html.Div([
                dcc.Graph(
                    id='hourly-heatmap',
                    config={'displayModeBar': False},
                    style={'height': '220px'}
                )
            ], style=heatmap_style),

            html.Div([
                dcc.Graph(
                    id='seasonal-heatmap',
                    config={'displayModeBar': False},
                    style={'height': '220px'}
                )
            ], style=heatmap_style)

        ], style={
            'flex': '3',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '8px'
        }),

        # Map
        html.Div([
            html.H4("WHERE IS DEMAND HIGHEST?", style={'margin': '2px 0'}),
            html.H5("Pickup Density (based on selection)", style={'marginBottom': '6px'}),
            html.Iframe(
                id='map',
                srcDoc='',
                width='100%',
                height='250'
            )
        ], style={
            'flex': '2',
            'background': 'white',
            'borderRadius': '10px',
            'padding': '10px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.08)'
        }),

    ], style={
        'display': 'flex',
        'width': '100%',
        'gap': '8px',
        'marginTop': '8px'
    }),

    html.Div(
        id='top-pickup-zones',
        style={
            'marginTop': '8px',
            'fontSize': '12px'
        }
    )
])

temporal_layout = html.Div([
    html.H2("Temporal Analysis"),
    html.Div("xxx"),
    html.Div("xxx")
])

spatial_layout = html.Div([
    html.H2("Spatial Analysis"),
    html.Div("yyy"),
    html.Div("yyy")
])

characteristics_layout = html.Div([
    html.H2("Trip Characteristics"),
    html.Div(
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))',
            'gap': '8px',
            'overflow': 'auto'
        },
        children=[
            dcc.Graph(id='char-time-series', config={'displayModeBar': False}),
            dcc.Graph(id='char-fare-distance', config={'displayModeBar': False}),
            dcc.Graph(id='char-tip-distance', config={'displayModeBar': False}),
            dcc.Graph(id='char-revenue-over-time', config={'displayModeBar': False}),
            dcc.Graph(id='char-passenger-count', config={'displayModeBar': False}),
            dcc.Graph(id='char-payment-donut', config={'displayModeBar': False}),
        ]
    )
])

app.layout = html.Div([
    dcc.Store(id='selected-hour-weekdays', data=[]),
    dcc.Store(id='selected-month-weekdays', data=[]),

    # LEFT SIDEBAR
    html.Div([
        html.H2("🚕 NYC Taxi Dashboard", style={
            'color': 'white',
            'fontSize': '18px',
            'marginBottom': '25px'
        }),

        html.H4("Filters", style={
            'color': '#f1c40f',
            'marginBottom': '10px'
        }),

        filter_bar,

        html.Div(
            id='active-time-selection',
            style={'display': 'none'}
        )
    ], style=sidebar_style),

    # TOP NAVIGATION
    html.Div([
        html.Button("Overview", id='nav-overview', n_clicks=0, style=active_nav_style),
        html.Button("Temporal Analysis", id='nav-temporal', n_clicks=0, style=inactive_nav_style),
        html.Button("Spatial Analysis", id='nav-spatial', n_clicks=0, style=inactive_nav_style),
        html.Button("Trip Characteristics", id='nav-characteristics', n_clicks=0, style=inactive_nav_style),
    ], style=main_content_style),

    # RIGHT MAIN CONTENT
    html.Div([
        html.Div(id='overview-page', children=overview_layout, style={'display': 'block', 'width': '100%'}),
        html.Div(id='temporal-page', children=temporal_layout, style={'display': 'none', 'width': '100%'}),
        html.Div(id='spatial-page', children=spatial_layout, style={'display': 'none', 'width': '100%'}),
        html.Div(id='characteristics-page', children=characteristics_layout, style={'display': 'none', 'width': '100%'})
    ], style=main_content_style)
], style={
    'fontFamily': '"Segoe UI", sans-serif'
})

# ------------------------------
# 4. Callbacks
# ------------------------------
@app.callback(
    [
        Output('date-range', 'start_date'),
        Output('date-range', 'end_date'),
        Output('pickup-borough-dropdown', 'value'),
        Output('dropoff-borough-dropdown', 'value'),
        Output('payment-dropdown', 'value', allow_duplicate=True),
        Output('passenger-range', 'value'),
        Output('selected-hour-weekdays', 'data'),
        Output('selected-month-weekdays', 'data')
    ],
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return (
        min_date,
        max_date,
        'All',
        'All',
        'All',
        [0, max_passengers],
        [],
        []
    )

@app.callback(
    Output('selected-hour-weekdays', 'data', allow_duplicate=True),
    Input('hourly-heatmap', 'clickData'),
    State('selected-hour-weekdays', 'data'),
    prevent_initial_call=True
)
def toggle_hour_weekday(clickData, selected_cells):
    if clickData is None:
        return no_update

    point = clickData['points'][0]

    cell = {
        'hour': int(point['x']),
        'weekday': int(point['y'])
    }

    if cell in selected_cells:
        selected_cells.remove(cell)
    else:
        selected_cells.append(cell)

    return selected_cells

@app.callback(
    Output('selected-month-weekdays', 'data', allow_duplicate=True),
    Input('seasonal-heatmap', 'clickData'),
    State('selected-month-weekdays', 'data'),
    prevent_initial_call=True
)
def toggle_month_weekday(clickData, selected_cells):
    if clickData is None:
        return no_update

    point = clickData['points'][0]

    cell = {
        'month': int(point['y']),
        'weekday': int(point['x'])
    }

    if cell in selected_cells:
        selected_cells.remove(cell)
    else:
        selected_cells.append(cell)

    return selected_cells

def compare_kpi(title, baseline_value, selected_value, suffix="", prefix="", decimals=1, comparison_type="average"):
    if comparison_type == "total":
        share_pct = (
            (selected_value / baseline_value) * 100
            if baseline_value > 0 else 0
        )

        comparison_text = f"{share_pct:.1f}% of filtered total"
        comparison_color = '#2980b9'
        baseline_label = "Filtered total"

    else:
        diff_pct = (
            ((selected_value - baseline_value) / baseline_value) * 100
            if baseline_value != 0 else 0
        )

        diff_sign = '+' if diff_pct >= 0 else ''
        comparison_text = f"{diff_sign}{diff_pct:.1f}% vs filtered avg"
        comparison_color = 'green' if diff_pct >= 0 else 'red'
        baseline_label = "Filtered avg"

    return html.Div([
        html.H4(title, style={'margin': '0', 'color': '#666'}),

        html.Div([
            html.Div(baseline_label, style={'fontSize': '11px', 'color': '#999'}),
            html.H4(
                f"{prefix}{baseline_value:,.{decimals}f}{suffix}",
                style={'marginBottom': '8px', 'color': '#555'}
            )
        ]),

        html.Div([
            html.Div("Selection", style={'fontSize': '11px', 'color': '#999'}),
            html.H3(
                f"{prefix}{selected_value:,.{decimals}f}{suffix}",
                style={'margin': '2px 0 0 0', 'color': 'black'}
            ),
            html.Div(
                comparison_text,
                style={
                    'color': comparison_color,
                    'fontWeight': 'bold',
                    'fontSize': '12px'
                }
            )
        ])
    ])

@app.callback(
    [
        Output('total-trips', 'children'),
        Output('total-revenue', 'children'),
        Output('avg-fare', 'children'),
        Output('avg-tip-pct', 'children'),
        Output('avg-distance', 'children'),
        Output('avg-duration', 'children'),
        Output('hourly-heatmap', 'figure'),
        Output('seasonal-heatmap', 'figure'),
        Output('map', 'srcDoc'),
        Output('top-pickup-zones', 'children'),
        Output('active-time-selection', 'children'),
        Output('active-time-selection', 'style'),
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('pickup-borough-dropdown', 'value'),
        Input('dropoff-borough-dropdown', 'value'),
        Input('payment-dropdown', 'value'),
        Input('passenger-range', 'value'),
        Input('selected-hour-weekdays', 'data'),
        Input('selected-month-weekdays', 'data')
    ]
)
def update_kpis(
        start_date,
        end_date,
        selected_pickup_borough,
        selected_dropoff_borough,
        selected_payment,
        selected_passenger_range,
        selected_hour_weekdays,
        selected_month_weekdays
    ):
    base_filtered = get_filtered_trips(
        start_date,
        end_date,
        selected_pickup_borough,
        selected_dropoff_borough,
        selected_payment,
        selected_passenger_range
    )

    selected_filtered = get_filtered_trips(
        start_date,
        end_date,
        selected_pickup_borough,
        selected_dropoff_borough,
        selected_payment,
        selected_passenger_range,
        selected_hour_weekdays,
        selected_month_weekdays
    )

    # Empty data handling
    if len(selected_filtered) == 0:
        empty_card = html.Div([html.H4("No data", style={'margin': '0'}), html.H3("0", style={'margin': '0'})])
        empty_map = "<div style='text-align:center; padding:20px;'>No data</div>"
        empty_top_zones = html.Div("No top zones", style={'fontSize': '12px', 'color': '#999'})
        active_style = {'display': 'none'}
        return (
            empty_card,
            empty_card,
            empty_card,
            empty_card,
            empty_card,
            empty_card,
            empty_figure("No data"),
            empty_figure("No data"),
            empty_map,
            empty_top_zones,
            "",
            active_style
        )

    # KPI values
    baseline_total_trips = len(base_filtered)
    baseline_total_revenue = base_filtered['total_amount'].sum()
    baseline_avg_fare = base_filtered['fare_amount'].mean()
    baseline_avg_tip_pct = base_filtered['tip_percentage'].mean()
    baseline_avg_distance = base_filtered['trip_distance'].mean()
    baseline_avg_duration = base_filtered['trip_duration_min'].mean()

    selected_total_trips = len(selected_filtered)
    selected_total_revenue = selected_filtered['total_amount'].sum()
    selected_avg_fare = selected_filtered['fare_amount'].mean()
    selected_avg_tip_pct = selected_filtered['tip_percentage'].mean()
    selected_avg_distance = selected_filtered['trip_distance'].mean()
    selected_avg_duration = selected_filtered['trip_duration_min'].mean()

    total_card = compare_kpi("Total Trips", baseline_total_trips, selected_total_trips, decimals=0, comparison_type="total")
    revenue_card = compare_kpi("Total Revenue", baseline_total_revenue, selected_total_revenue, prefix="$", decimals=0, comparison_type="total")
    fare_card = compare_kpi("Avg Fare", baseline_avg_fare, selected_avg_fare, prefix="$", decimals=2)
    tip_card = compare_kpi("Avg Tip %", baseline_avg_tip_pct, selected_avg_tip_pct, suffix="%", decimals=1)
    dist_card = compare_kpi("Avg Distance", baseline_avg_distance, selected_avg_distance, suffix=" mi", decimals=1)
    duration_card = compare_kpi("Avg Duration", baseline_avg_duration, selected_avg_duration, suffix=" min", decimals=1)

    if len(base_filtered) > 0:
        heatmap_data = (
            base_filtered
            .groupby(['pickup_hour', 'pickup_dayofweek'])
            .size()
            .reset_index(name='count')
        )

        heatmap_pivot = heatmap_data.pivot(
            index='pickup_dayofweek',
            columns='pickup_hour',
            values='count'
        ).fillna(0)

        for hour in range(24):
            if hour not in heatmap_pivot.columns:
                heatmap_pivot[hour] = 0

        for day in range(7):
            if day not in heatmap_pivot.index:
                heatmap_pivot.loc[day] = 0

        heatmap_pivot = heatmap_pivot.reindex(
            index=range(7),
            columns=range(24),
            fill_value=0
        )

        day_labels_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Trips"),
            title="Trip Volume Heatmap (Hour vs. Day)",
            color_continuous_scale='Plasma',
            aspect='auto'
        )

        fig_heatmap.update_yaxes(
            tickvals=list(range(7)),
            ticktext=day_labels_short
        )

        fig_heatmap.update_layout(height=220, margin=dict(l=20, r=10, t=30, b=20))
        fig_heatmap = apply_style_to_figure(fig_heatmap)

        for cell in selected_hour_weekdays:
            fig_heatmap.add_shape(
                type="rect",
                x0=cell['hour'] - 0.5,
                x1=cell['hour'] + 0.5,
                y0=cell['weekday'] - 0.5,
                y1=cell['weekday'] + 0.5,
                line=dict(color="black", width=3),
                fillcolor="rgba(0,0,0,0)"
            )

    else:
        fig_heatmap = empty_figure("No data")

    if len(base_filtered) > 0:
        daily_counts = (
            base_filtered
            .groupby(['pickup_date', 'pickup_month', 'pickup_dayofweek'])
            .size()
            .reset_index(name='trips')
        )

        heatmap2_data = (
            daily_counts
            .groupby(['pickup_month', 'pickup_dayofweek'])['trips']
            .mean()
            .reset_index()
        )

        heatmap2_pivot = heatmap2_data.pivot(
            index='pickup_month',
            columns='pickup_dayofweek',
            values='trips'
        )

        fig_heatmap2 = px.imshow(
            heatmap2_pivot,
            labels=dict(x="Day of Week", y="Month of Year", color="Trips"),
            title="Trip Volume Heatmap (Day vs. Month)",
            color_continuous_scale='Plasma',
            aspect='auto'
        )

        month_labels = [
            'Jan', 'Feb', 'Mar', 'Apr',
            'May', 'Jun', 'Jul', 'Aug',
            'Sep', 'Oct', 'Nov', 'Dec'
        ]

        day_labels_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig_heatmap2.update_yaxes(
            tickvals=list(range(1, 13)),
            ticktext=month_labels
        )

        fig_heatmap2.update_xaxes(
            tickvals=list(range(7)),
            ticktext=day_labels_short
        )

        fig_heatmap2.update_layout(height=220, margin=dict(l=20, r=10, t=30, b=20))
        fig_heatmap2 = apply_style_to_figure(fig_heatmap2)

        for cell in selected_month_weekdays:
            fig_heatmap2.add_shape(
                type="rect",
                x0=cell['weekday'] - 0.5,
                x1=cell['weekday'] + 0.5,
                y0=cell['month'] - 0.5,
                y1=cell['month'] + 0.5,
                line=dict(color="black", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    else:
        fig_heatmap2 = empty_figure("No data")

    pickup_counts = selected_filtered['PULocationID'].value_counts().reset_index()
    pickup_counts.columns = ['LocationID', 'count']

    zones_with_counts = zones_gdf.merge(
        pickup_counts,
        on='LocationID',
        how='left'
    ).fillna(0)

    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    folium.Choropleth(
        geo_data=json.loads(zones_with_counts.to_json()),
        data=zones_with_counts,
        columns=['LocationID', 'count'],
        key_on='feature.properties.LocationID',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Pickups',
        highlight=True
    ).add_to(m)

    map_html = m.get_root().render()

    top_zones = (
        selected_filtered
        .groupby('PULocationID')
        .size()
        .reset_index(name='trips')
        .merge(
            zones_gdf[['LocationID', 'zone', 'borough']],
            left_on='PULocationID',
            right_on='LocationID',
            how='left'
        )
        .sort_values('trips', ascending=False)
        .head(5)
    )

    top_zones_list = html.Div([
        html.H5("Top 5 pick-up zones (in selection)", style={'margin': '6px 0'}),

        html.Ol([
            html.Li(
                f"{row['zone']} ({row['borough']}) — {row['trips']:,} trips"
            )
            for _, row in top_zones.iterrows()
        ], style={
            'margin': '0',
            'paddingLeft': '18px'
        })
    ])

    day_labels_full = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday'
    ]
    month_labels_full = [
        None,
        'January', 'February', 'March', 'April',
        'May', 'June', 'July', 'August',
        'September', 'October', 'November', 'December'
    ]

    selection_parts = []

    for cell in selected_hour_weekdays:
        selection_parts.append(
            f"{day_labels_full[int(cell['weekday'])]} "
            f"{int(cell['hour']):02d}:00–{int(cell['hour']) + 1:02d}:00"
        )

    for cell in selected_month_weekdays:
        selection_parts.append(
            f"{month_labels_full[int(cell['month'])]} "
            f"{day_labels_full[int(cell['weekday'])]}"
        )

    if selection_parts:
        active_text = html.Div([
            html.Strong("Active selections: "),
            html.Br(),
            html.Ul([html.Li(part) for part in selection_parts])
        ])

        active_style = {
            'background': '#fff7d6',
            'border': '1px solid #f1c40f',
            'borderRadius': '8px',
            'padding': '10px',
            'marginBottom': '12px',
            'display': 'block'
        }
    else:
        active_text = ""
        active_style = {'display': 'none'}

    return (
        total_card,
        revenue_card,
        fare_card,
        tip_card,
        dist_card,
        duration_card,
        fig_heatmap,
        fig_heatmap2,
        map_html,
        top_zones_list,
        active_text,
        active_style
    )

@app.callback(
    [
        Output('char-time-series', 'figure'),
        Output('char-fare-distance', 'figure'),
        Output('char-tip-distance', 'figure'),
        Output('char-revenue-over-time', 'figure'),
        Output('char-passenger-count', 'figure'),
        Output('char-payment-donut', 'figure')
    ],
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('pickup-borough-dropdown', 'value'),
        Input('dropoff-borough-dropdown', 'value'),
        Input('payment-dropdown', 'value'),
        Input('passenger-range', 'value'),
        Input('selected-hour-weekdays', 'data'),
        Input('selected-month-weekdays', 'data')
    ]
)
def update_characteristics_charts(
    start_date,
    end_date,
    selected_pickup_borough,
    selected_dropoff_borough,
    selected_payment,
    selected_passenger_range,
    selected_hour_weekdays,
    selected_month_weekdays
):
    filtered = get_filtered_trips(
        start_date,
        end_date,
        selected_pickup_borough,
        selected_dropoff_borough,
        selected_payment,
        selected_passenger_range,
        selected_hour_weekdays,
        selected_month_weekdays
    )

    if len(filtered) == 0:
        empty_fig = empty_figure("No data")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    daily = (
        filtered
        .groupby(filtered['tpep_pickup_datetime'].dt.date)
        .size()
        .reset_index(name='trips')
    )
    daily.columns = ['date', 'trips']

    fig_time = px.line(
        daily,
        x='date',
        y='trips',
        title='Trips Over Time'
    )
    fig_time.update_layout(height=230)
    fig_time = apply_style_to_figure(fig_time)

    fare_sample = filtered.sample(min(3000, len(filtered)))
    fig_fare_dist = px.scatter(
        fare_sample,
        x='trip_distance',
        y='fare_amount',
        title='Fare vs. Distance',
        labels={'trip_distance': 'Miles', 'fare_amount': '$'},
        opacity=0.5,
        trendline='ols'
    )
    fig_fare_dist.update_layout(height=230)
    fig_fare_dist = apply_style_to_figure(fig_fare_dist)

    tip_sample = filtered.sample(min(3000, len(filtered)))
    fig_tip_dist = px.scatter(
        tip_sample,
        x='trip_distance',
        y='tip_percentage',
        title='Tip % vs. Distance',
        labels={'trip_distance': 'Miles', 'tip_percentage': 'Tip %'},
        opacity=0.5,
        trendline='lowess'
    )
    fig_tip_dist.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_tip_dist.update_layout(height=230)
    fig_tip_dist = apply_style_to_figure(fig_tip_dist)

    revenue_daily = (
        filtered
        .groupby(filtered['tpep_pickup_datetime'].dt.date)['total_amount']
        .sum()
        .reset_index()
    )

    revenue_daily.columns = ['date', 'revenue']

    fig_revenue = px.line(
        revenue_daily,
        x='date',
        y='revenue',
        title='Revenue Over Time',
        labels={'date': 'Date', 'revenue': 'Revenue ($)'}
    )
    fig_revenue.update_layout(height=230)
    fig_revenue = apply_style_to_figure(fig_revenue)

    passenger_counts = (
        filtered['passenger_count']
        .value_counts()
        .reset_index()
    )
    passenger_counts.columns = ['passengers', 'count']
    passenger_counts = passenger_counts.sort_values('passengers')

    fig_passenger = px.bar(
        passenger_counts,
        x='passengers',
        y='count',
        title='Trips by Passenger Count',
        labels={'passengers': 'Passengers', 'count': 'Trips'}
    )
    fig_passenger.update_layout(height=230)
    fig_passenger = apply_style_to_figure(fig_passenger)

    payment_counts = (
        filtered['payment_type_name']
        .value_counts()
        .reset_index()
    )
    payment_counts.columns = ['method', 'count']

    fig_payment = px.pie(
        payment_counts,
        names='method',
        values='count',
        title='Payment Methods',
        hole=0.4
    )
    fig_payment.update_layout(height=230)
    fig_payment = apply_style_to_figure(fig_payment)

    return (
        fig_time,
        fig_fare_dist,
        fig_tip_dist,
        fig_revenue,
        fig_passenger,
        fig_payment
    )

@app.callback(
    [
        Output('nav-overview', 'style'),
        Output('nav-temporal', 'style'),
        Output('nav-spatial', 'style'),
        Output('nav-characteristics', 'style'),
        Output('overview-page', 'style'),
        Output('temporal-page', 'style'),
        Output('spatial-page', 'style'),
        Output('characteristics-page', 'style')
    ],
    [
        Input('nav-overview', 'n_clicks'),
        Input('nav-temporal', 'n_clicks'),
        Input('nav-spatial', 'n_clicks'),
        Input('nav-characteristics', 'n_clicks')
    ],
    prevent_initial_call=True
)
def switch_view(n_overview, n_temporal, n_spatial, n_characteristics):

    triggered = ctx.triggered_id

    current_view = 'overview'

    if triggered == 'nav-temporal':
        current_view = 'temporal'
    elif triggered == 'nav-spatial':
        current_view = 'spatial'
    elif triggered == 'nav-characteristics':
        current_view = 'characteristics'

    styles = {
        'overview': active_nav_style if current_view == 'overview' else inactive_nav_style,
        'temporal': active_nav_style if current_view == 'temporal' else inactive_nav_style,
        'spatial': active_nav_style if current_view == 'spatial' else inactive_nav_style,
        'characteristics': active_nav_style if current_view == 'characteristics' else inactive_nav_style
    }

    visible = {'display': 'block', 'width': '100%'}
    hidden = {'display': 'none', 'width': '100%'}

    return (
        styles['overview'],
        styles['temporal'],
        styles['spatial'],
        styles['characteristics'],
        visible if current_view == 'overview' else hidden,
        visible if current_view == 'temporal' else hidden,
        visible if current_view == 'spatial' else hidden,
        visible if current_view == 'characteristics' else hidden
    )

# ------------------------------
# 5. Run the app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)