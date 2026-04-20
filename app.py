# app.py (KPI cards: light gray background, black text)
import pandas as pd
import geopandas as gpd
import folium
import json
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np

# Helper function for empty charts
def empty_figure(message="No data"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=30))
    return fig

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

# Get min and max dates for the date picker
min_date = trips_df['tpep_pickup_datetime'].min().date()
max_date = trips_df['tpep_pickup_datetime'].max().date()

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

# ------------------------------
# 2. Initialize Dash app
# ------------------------------
app = Dash(__name__)
server = app.server

# ------------------------------
# 3. App layout – compact, no scroll
# ------------------------------
app.layout = html.Div(
    style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column', 'margin': '0', 'padding': '0 10px', 'boxSizing': 'border-box'},
    children=[
        html.H1("🗽 NYC Yellow Taxi Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '5px 0', 'fontSize': '24px'}),
        
        # Filter bar
        html.Div([
            html.Div([
                html.Label("Date Range:", style={'fontWeight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=min_date,
                    end_date=max_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format='YYYY-MM-DD',
                    style={'marginLeft': '10px', 'fontSize': '12px'}
                ),
            ], style={'width': '40%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Borough:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='borough-dropdown',
                    options=[{'label': 'All Boroughs', 'value': 'All'}] + 
                            [{'label': b, 'value': b} for b in zones_gdf['borough'].unique() if b != 'Other'],
                    value='All',
                    clearable=False,
                    style={'width': '180px', 'display': 'inline-block', 'marginLeft': '10px', 'fontSize': '12px'}
                ),
            ], style={'width': '35%', 'display': 'inline-block', 'float': 'right'}),
        ], style={'padding': '5px', 'borderBottom': '1px solid #ddd', 'marginBottom': '5px', 'flexShrink': '0'}),
        
        # KPI Row – light gray background, black text
        html.Div([
            html.Div(id='total-trips', style={'flex': '1', 'background': '#d3d3d3', 'color': 'black', 'borderRadius': '8px', 'padding': '5px', 'margin': '3px', 'textAlign': 'center'}),
            html.Div(id='avg-fare', style={'flex': '1', 'background': '#d3d3d3', 'color': 'black', 'borderRadius': '8px', 'padding': '5px', 'margin': '3px', 'textAlign': 'center'}),
            html.Div(id='avg-distance', style={'flex': '1', 'background': '#d3d3d3', 'color': 'black', 'borderRadius': '8px', 'padding': '5px', 'margin': '3px', 'textAlign': 'center'}),
            html.Div(id='avg-tip-pct', style={'flex': '1', 'background': '#d3d3d3', 'color': 'black', 'borderRadius': '8px', 'padding': '5px', 'margin': '3px', 'textAlign': 'center'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'marginBottom': '5px', 'flexShrink': '0'}),
        
        # Map
        html.Div([
            html.H4("📍 Pickup Volume by Taxi Zone", style={'textAlign': 'center', 'margin': '3px'}),
            html.Iframe(id='map', srcDoc='', width='100%', height='220')
        ], style={'marginBottom': '5px', 'flexShrink': '0'}),
        
        # Charts grid (2 rows, 3 columns)
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr 1fr',
                'gridAutoRows': 'minmax(230px, auto)',
                'gap': '8px',
                'flex': '1',
                'minHeight': '0',
                'overflow': 'auto'
            },
            children=[
                dcc.Graph(id='time-series', config={'displayModeBar': False}, style={'height': '230px'}),
                dcc.Graph(id='fare-distance', config={'displayModeBar': False}, style={'height': '230px'}),
                dcc.Graph(id='tip-distance', config={'displayModeBar': False}, style={'height': '230px'}),
                dcc.Graph(id='hourly-heatmap', config={'displayModeBar': False}, style={'height': '230px'}),
                dcc.Graph(id='passenger-count', config={'displayModeBar': False}, style={'height': '230px'}),
                dcc.Graph(id='payment-donut', config={'displayModeBar': False}, style={'height': '230px'}),
            ]
        ),
    ]
)

# ------------------------------
# 4. Main callback
# ------------------------------
@app.callback(
    [Output('total-trips', 'children'),
     Output('avg-fare', 'children'),
     Output('avg-distance', 'children'),
     Output('avg-tip-pct', 'children'),
     Output('map', 'srcDoc'),
     Output('time-series', 'figure'),
     Output('fare-distance', 'figure'),
     Output('tip-distance', 'figure'),
     Output('hourly-heatmap', 'figure'),
     Output('passenger-count', 'figure'),
     Output('payment-donut', 'figure')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('borough-dropdown', 'value'),
     Input('payment-donut', 'clickData')]
)
def update_dashboard(start_date, end_date, selected_borough, payment_click):
    # Handle None dates
    if start_date is None:
        start_date = min_date
    if end_date is None:
        end_date = max_date
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    filtered = trips_df[
        (trips_df['tpep_pickup_datetime'] >= start) &
        (trips_df['tpep_pickup_datetime'] <= end)
    ].copy()
    
    # Filter by borough
    if selected_borough != 'All':
        filtered = filtered[filtered['PULocationBorough'] == selected_borough]
    
    # Cross-filter by payment method
    if payment_click and 'points' in payment_click and len(payment_click['points']) > 0:
        clicked_payment = payment_click['points'][0]['label']
        filtered = filtered[filtered['payment_type_name'] == clicked_payment]
    
    # Empty data handling
    if len(filtered) == 0:
        empty_kpi = html.Div([html.H4("No data", style={'margin': '0'}), html.H3("0", style={'margin': '0'})])
        empty_map = "<div style='text-align:center; padding:20px;'>No data</div>"
        empty_fig = empty_figure("No data")
        return (empty_kpi, empty_kpi, empty_kpi, empty_kpi,
                empty_map, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig)
    
    # KPI values
    total = len(filtered)
    avg_fare = filtered['fare_amount'].mean()
    avg_dist = filtered['trip_distance'].mean()
    avg_tip_pct = filtered['tip_percentage'].mean()
    
    total_card = html.Div([html.H4("Total Trips", style={'margin': '0', 'color': 'black'}), html.H3(f"{total:,}", style={'margin': '0', 'color': 'black'})])
    fare_card = html.Div([html.H4("Avg Fare", style={'margin': '0', 'color': 'black'}), html.H3(f"${avg_fare:.2f}", style={'margin': '0', 'color': 'black'})])
    dist_card = html.Div([html.H4("Avg Distance", style={'margin': '0', 'color': 'black'}), html.H3(f"{avg_dist:.1f} mi", style={'margin': '0', 'color': 'black'})])
    tip_card = html.Div([html.H4("Avg Tip %", style={'margin': '0', 'color': 'black'}), html.H3(f"{avg_tip_pct:.1f}%", style={'margin': '0', 'color': 'black'})])
    
    # Map
    pickup_counts = filtered['PULocationID'].value_counts().reset_index()
    pickup_counts.columns = ['LocationID', 'count']
    zones_with_counts = zones_gdf.merge(pickup_counts, on='LocationID', how='left').fillna(0)
    
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB positron')
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
    
    # Time series
    daily = filtered.groupby(filtered['tpep_pickup_datetime'].dt.date).size().reset_index()
    daily.columns = ['date', 'trips']
    fig_time = px.line(daily, x='date', y='trips', title='Trips Over Time')
    fig_time.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    
    # Fare vs Distance
    if len(filtered) > 0:
        fare_sample = filtered.sample(min(3000, len(filtered)))
        fig_fare_dist = px.scatter(fare_sample, x='trip_distance', y='fare_amount',
                                   title='Fare vs. Distance',
                                   labels={'trip_distance': 'Miles', 'fare_amount': '$'},
                                   opacity=0.5, trendline='ols')
        fig_fare_dist.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    else:
        fig_fare_dist = empty_figure("No data")
    
    # Tip % vs Distance
    if len(filtered) > 0:
        tip_sample = filtered.sample(min(3000, len(filtered)))
        fig_tip_dist = px.scatter(tip_sample, x='trip_distance', y='tip_percentage',
                                  title='Tip % vs. Distance (0% = no tip)',
                                  labels={'trip_distance': 'Miles', 'tip_percentage': 'Tip %'},
                                  opacity=0.5, trendline='lowess')
        fig_tip_dist.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_tip_dist.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    else:
        fig_tip_dist = empty_figure("No data")
    
    # Heatmap (flipped: hour on X, day on Y)
    if len(filtered) > 0:
        heatmap_data = filtered.groupby(['pickup_hour', 'pickup_dayofweek']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='pickup_dayofweek', columns='pickup_hour', values='count').fillna(0)
        for hour in range(24):
            if hour not in heatmap_pivot.columns:
                heatmap_pivot[hour] = 0
        for day in range(7):
            if day not in heatmap_pivot.index:
                heatmap_pivot.loc[day] = 0
        heatmap_pivot = heatmap_pivot.reindex(index=range(7), columns=range(24), fill_value=0)
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_heatmap = px.imshow(heatmap_pivot,
                                labels=dict(x="Hour of Day", y="Day of Week", color="Trips"),
                                title="Trip Volume Heatmap (Hour vs. Day)",
                                color_continuous_scale='Plasma',
                                aspect='auto')
        fig_heatmap.update_yaxes(tickvals=list(range(7)), ticktext=day_labels)
        fig_heatmap.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    else:
        fig_heatmap = empty_figure("No data")
    
    # Passenger count
    passenger_counts = filtered['passenger_count'].value_counts().reset_index()
    passenger_counts.columns = ['passengers', 'count']
    passenger_counts = passenger_counts.sort_values('passengers')
    fig_passenger = px.bar(passenger_counts, x='passengers', y='count',
                           title='Trips by Passenger Count',
                           labels={'passengers': 'Passengers', 'count': 'Trips'})
    fig_passenger.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    
    # Payment donut
    payment_counts = filtered['payment_type_name'].value_counts().reset_index()
    payment_counts.columns = ['method', 'count']
    fig_payment = px.pie(payment_counts, names='method', values='count',
                         title='Payment Methods', hole=0.4)
    fig_payment.update_layout(height=230, margin=dict(l=30, r=20, t=40, b=30))
    
    return (total_card, fare_card, dist_card, tip_card,
            map_html, fig_time, fig_fare_dist, fig_tip_dist,
            fig_heatmap, fig_passenger, fig_payment)

# ------------------------------
# 5. Run the app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)