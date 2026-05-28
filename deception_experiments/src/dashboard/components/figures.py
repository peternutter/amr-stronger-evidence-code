"""Plotly figure builders for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def create_line_chart_figure(
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 500,
    show_legend: bool = True,
) -> go.Figure:
    """Create a standardized Plotly line chart figure.

    Args:
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        show_legend: Whether to show the legend

    Returns:
        Configured Plotly Figure object
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        height=height,
        showlegend=show_legend,
        hovermode="x unified",
    )
    return fig


def create_heatmap_figure(
    z: list[list[float]],
    x: list[str],
    y: list[str],
    title: str,
    text: list[list[str]] | None = None,
    colorscale: str = "RdBu_r",
    height: int = 600,
    width: int | None = None,
    show_colorbar: bool = True,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Create a standardized Plotly heatmap figure.

    Args:
        z: 2D array of values
        x: X-axis labels
        y: Y-axis labels
        title: Chart title
        text: Optional 2D array of text annotations
        colorscale: Plotly colorscale name
        height: Chart height in pixels
        width: Chart width in pixels (None for auto)
        show_colorbar: Whether to show the color scale
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale

    Returns:
        Configured Plotly Figure object
    """
    heatmap_kwargs: dict[str, Any] = {
        "z": z,
        "x": x,
        "y": y,
        "colorscale": colorscale,
        "showscale": show_colorbar,
        "hovertemplate": "X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>",
    }

    if text is not None:
        heatmap_kwargs["text"] = text
        heatmap_kwargs["texttemplate"] = "%{text}"

    if zmin is not None:
        heatmap_kwargs["zmin"] = zmin
    if zmax is not None:
        heatmap_kwargs["zmax"] = zmax

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    layout_kwargs: dict[str, Any] = {
        "title": title,
        "template": "plotly_dark",
        "height": height,
    }
    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)
    return fig


def create_confusion_matrix_figure(
    cm: list[list[int]],
    labels: list[str],
    height: int = 400,
    width: int = 450,
) -> go.Figure:
    """Create a confusion matrix heatmap figure.

    Args:
        cm: 2D confusion matrix
        labels: Class labels
        height: Chart height in pixels
        width: Chart width in pixels

    Returns:
        Configured Plotly Figure object
    """
    # Calculate total for percentages
    total = sum(sum(row) for row in cm)

    # Create text annotations with counts and percentages
    cm_text = []
    for i in range(len(cm)):
        row_text = []
        for j in range(len(cm[i])):
            count = cm[i][j]
            pct = 100.0 * count / total if total else 0.0
            row_text.append(f"{count}<br>({pct:.1f}%)")
        cm_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm_text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale="Blues",
            showscale=True,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="True Label", autorange="reversed"),
        template="plotly_dark",
        height=height,
        width=width,
    )

    return fig


def create_roc_curve_figure(
    fpr: list[float],
    tpr: list[float],
    auc: float,
    height: int = 350,
    show_legend: bool = True,
) -> go.Figure:
    """Create an ROC curve figure.

    Args:
        fpr: False positive rate values
        tpr: True positive rate values
        auc: Area under the curve value
        height: Chart height in pixels
        show_legend: Whether to show the legend

    Returns:
        Configured Plotly Figure object
    """
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc:.4f})",
            line=dict(color="#e94560", width=2),
        )
    )

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        height=height,
        showlegend=show_legend,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )

    return fig
