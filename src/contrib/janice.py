"""Causal leader-flow features contributed by Janice."""

from __future__ import annotations

import pandas as pd


def leader_flow_impulse(
    df: pd.DataFrame, span: int = 12
) -> pd.Series:
    """Return smoothed BTC/ETH signed flow, aligned with df.index.
    """
    if (
        not isinstance(span, int)
        or isinstance(span, bool)
        or span < 1
    ):
        raise ValueError("span must be a positive integer")

    if df.empty:
        return pd.Series(
            index=df.index,
            dtype=float,
            name="leader_flow_impulse",
        )

    leaders = df.loc[
        df["symbol"].isin(["BTCUSDT", "ETHUSDT"])
    ].copy()

    counts = (
        leaders.groupby(["timestamp", "symbol"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(
        index=pd.Index(df["timestamp"].unique()),
        columns=["BTCUSDT", "ETHUSDT"],
        fill_value=0,
    )
    if not counts.eq(1).all().all():
        raise ValueError(
            "Each timestamp requires one BTC and one ETH row"
        )

    leaders["signed_quote"] = (
        2.0 * leaders["taker_buy_quote_volume"]
        - leaders["quote_volume"]
    )

    hourly = leaders.groupby("timestamp", sort=True)[
        ["signed_quote", "quote_volume"]
    ].sum()

    raw = hourly["signed_quote"].div(
        hourly["quote_volume"].where(
            hourly["quote_volume"] > 0
        )
    ).fillna(0.0)

    smooth = raw.ewm(
        span=span,
        adjust=True,
        min_periods=1,
    ).mean()

    return (
        df["timestamp"]
        .map(smooth)
        .rename("leader_flow_impulse")
    )