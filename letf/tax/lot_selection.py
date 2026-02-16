from typing import List, Dict, Optional
from letf.tax.engine import LotSelectionMethod


def select_lot_fifo(positions: List[Dict], shares_needed: float) -> List[int]:
    """FIFO: Select lots starting from the oldest (first) lot."""
    selected = []
    remaining = shares_needed
    for i, pos in enumerate(positions):
        if remaining <= 0.001:
            break
        if pos['shares'] > 0.001:
            selected.append(i)
            remaining -= pos['shares']
    return selected


def select_lot_lifo(positions: List[Dict], shares_needed: float) -> List[int]:
    """LIFO: Select lots starting from the newest (last) lot."""
    selected = []
    remaining = shares_needed
    for i in range(len(positions) - 1, -1, -1):
        if remaining <= 0.001:
            break
        if positions[i]['shares'] > 0.001:
            selected.append(i)
            remaining -= positions[i]['shares']
    return selected


def select_lot_hifo(positions: List[Dict], shares_needed: float) -> List[int]:
    """HIFO: Highest In, First Out - Select lots with highest cost basis first."""
    lots_with_basis = [(i, pos['adjusted_price'])
                       for i, pos in enumerate(positions)
                       if pos['shares'] > 0.001]
    lots_with_basis.sort(key=lambda x: x[1], reverse=True)

    selected = []
    remaining = shares_needed
    for i, _ in lots_with_basis:
        if remaining <= 0.001:
            break
        selected.append(i)
        remaining -= positions[i]['shares']
    return selected


def select_lot_lofo(positions: List[Dict], shares_needed: float) -> List[int]:
    """LOFO: Lowest In, First Out - Select lots with lowest cost basis first."""
    lots_with_basis = [(i, pos['adjusted_price'])
                       for i, pos in enumerate(positions)
                       if pos['shares'] > 0.001]
    lots_with_basis.sort(key=lambda x: x[1])

    selected = []
    remaining = shares_needed
    for i, _ in lots_with_basis:
        if remaining <= 0.001:
            break
        selected.append(i)
        remaining -= positions[i]['shares']
    return selected


def select_lot_ltfo(positions: List[Dict], shares_needed: float,
                    sale_day: int, lt_threshold: int = 365) -> List[int]:
    """LTFO: Long-Term First Out - Select long-term lots first."""
    lt_lots = []
    st_lots = []

    for i, pos in enumerate(positions):
        if pos['shares'] <= 0.001:
            continue
        holding_days = sale_day - pos['original_day']
        if holding_days > lt_threshold:
            lt_lots.append((i, holding_days))
        else:
            st_lots.append((i, holding_days))

    lt_lots.sort(key=lambda x: x[1], reverse=True)
    st_lots.sort(key=lambda x: x[1], reverse=True)

    selected = []
    remaining = shares_needed

    for i, _ in lt_lots + st_lots:
        if remaining <= 0.001:
            break
        selected.append(i)
        remaining -= positions[i]['shares']
    return selected


def select_lot_stfo(positions: List[Dict], shares_needed: float,
                    sale_day: int, lt_threshold: int = 365) -> List[int]:
    """STFO: Short-Term First Out - Select short-term lots first."""
    st_lots = []
    lt_lots = []

    for i, pos in enumerate(positions):
        if pos['shares'] <= 0.001:
            continue
        holding_days = sale_day - pos['original_day']
        if holding_days > lt_threshold:
            lt_lots.append((i, holding_days))
        else:
            st_lots.append((i, holding_days))

    st_lots.sort(key=lambda x: x[1])
    lt_lots.sort(key=lambda x: x[1])

    selected = []
    remaining = shares_needed

    for i, _ in st_lots + lt_lots:
        if remaining <= 0.001:
            break
        selected.append(i)
        remaining -= positions[i]['shares']
    return selected


def select_lot_mintax(positions: List[Dict], shares_needed: float,
                      sale_day: int, sale_price: float,
                      lt_threshold: int = 365,
                      marginal_st_rate: float = 0.37,
                      marginal_lt_rate: float = 0.20) -> List[int]:
    """MINTAX: Algorithmically select lots to minimize tax."""
    lot_tax_impact = []

    for i, pos in enumerate(positions):
        if pos['shares'] <= 0.001:
            continue

        gain_per_share = sale_price - pos['adjusted_price']
        holding_days = sale_day - pos['original_day']
        is_lt = holding_days > lt_threshold

        if gain_per_share >= 0:
            tax_rate = marginal_lt_rate if is_lt else marginal_st_rate
        else:
            tax_rate = marginal_st_rate  # Losses offset highest rate gains first

        tax_impact_per_share = gain_per_share * tax_rate
        lot_tax_impact.append((i, tax_impact_per_share, pos['shares']))

    lot_tax_impact.sort(key=lambda x: x[1])  # Lowest tax impact first

    selected = []
    remaining = shares_needed
    for i, _, _ in lot_tax_impact:
        if remaining <= 0.001:
            break
        selected.append(i)
        remaining -= positions[i]['shares']
    return selected


def get_lots_to_sell(positions: List[Dict], shares_needed: float,
                     method: 'LotSelectionMethod', sale_day: int,
                     sale_price: float = None) -> List[int]:
    """Master function to select lots based on the chosen method."""
    if method == LotSelectionMethod.FIFO:
        return select_lot_fifo(positions, shares_needed)
    elif method == LotSelectionMethod.LIFO:
        return select_lot_lifo(positions, shares_needed)
    elif method == LotSelectionMethod.HIFO:
        return select_lot_hifo(positions, shares_needed)
    elif method == LotSelectionMethod.LOFO:
        return select_lot_lofo(positions, shares_needed)
    elif method == LotSelectionMethod.LTFO:
        return select_lot_ltfo(positions, shares_needed, sale_day)
    elif method == LotSelectionMethod.STFO:
        return select_lot_stfo(positions, shares_needed, sale_day)
    elif method in (LotSelectionMethod.MINTAX, LotSelectionMethod.SPEC_ID):
        if sale_price is None:
            return select_lot_hifo(positions, shares_needed)
        return select_lot_mintax(positions, shares_needed, sale_day, sale_price)
    else:
        return select_lot_fifo(positions, shares_needed)
