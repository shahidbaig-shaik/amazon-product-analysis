import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_aspect_radar(aspect_scores, product_name):
    df = pd.DataFrame(aspect_scores)
    product_data = df[df['product_name'] == product_name]
    
    if product_data.empty:
        return go.Figure()
        
    categories = product_data['aspect'].tolist()
    values = product_data['sentiment_score'].tolist()
    
    # Close the loop
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=product_name,
        line_color='#00CC96',
        fillcolor='rgba(0, 204, 150, 0.3)',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Aspect Sentiment Profile: {product_name}",
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1],
                gridcolor='rgba(128, 128, 128, 0.2)',
                linecolor='rgba(128, 128, 128, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                linecolor='rgba(128, 128, 128, 0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=14)
    )
    return fig

def plot_competitor_gap(competitor_gaps, target_product):
    df = pd.DataFrame(competitor_gaps)
    if df.empty:
        return None
        
    target_data = df[df['target_product'] == target_product].copy()
    if target_data.empty:
        return None
        
    # Bar chart showing gaps
    # Positive gap = Target is better (Green)
    # Negative gap = Competitor is better (Red)
    
    target_data['color'] = target_data['gap'].apply(lambda x: '#00CC96' if x > 0 else '#EF553B')
    
    fig = go.Figure()
    
    for competitor in target_data['competitor'].unique():
        subset = target_data[target_data['competitor'] == competitor]
        fig.add_trace(go.Bar(
            x=subset['aspect'],
            y=subset['gap'],
            name=competitor,
            marker_color=subset['color'],
            hovertemplate='<b>%{x}</b><br>Gap: %{y:.2f}<br>Competitor: ' + competitor + '<extra></extra>'
        ))

    fig.update_layout(
        title=f"Competitor Gaps vs {target_product}",
        yaxis_title="Sentiment Gap (Higher is Better)",
        xaxis_title="Aspect",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add a zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig
