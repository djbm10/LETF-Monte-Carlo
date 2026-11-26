"""Full-year resident wage + ordinary corporate investment-income subset.

MA DOR Form 1/Schedules B/D, 2023-25; MGL c.62 ss.2,4,5; DOR dated
surtax table 2024-26. No federal capital-loss wage deduction is imported.
Sources and exclusions: docs/integration-2026-11-24/TAX_AND_DATA.md.
"""
from math import isfinite
from letf.accuracy import UnsupportedModelError

SURTAX_THRESHOLDS = {2023: 1000000., 2024: 1053750., 2025: 1083150., 2026: 1107750.}
EXEMPTIONS = {'single': 4400., 'married': 8800., 'married_filing_jointly': 8800.,
              'head_of_household': 6800., 'married_filing_separately': 4400.}


def massachusetts_tax(year, ordinary_income, *, st_gains=0., st_losses=0.,
                      lt_gains=0., lt_losses=0., st_cf=0., lt_cf=0.,
                      interest=0., dividends=0., filing_status='single',
                      scope='full_year_resident_wages_and_corporate_securities'):
    if year not in SURTAX_THRESHOLDS or filing_status not in EXEMPTIONS:
        raise UnsupportedModelError('No enacted MA parameters for requested tax year/status')
    if scope != 'full_year_resident_wages_and_corporate_securities':
        raise UnsupportedModelError('MA pensions, nonresident allocation, special gains and business adjustments need separate rules')
    inputs = (ordinary_income, st_gains, st_losses, lt_gains, lt_losses, st_cf, lt_cf, interest, dividends)
    if any(not isfinite(v) or v < 0 for v in inputs):
        raise ValueError('MA input amounts must be finite and nonnegative')
    exemption = EXEMPTIONS[filing_status]
    if ordinary_income < exemption:
        raise UnsupportedModelError('MA excess-exemption allocation and no-tax-status/limited-income-credit worksheets are not implemented')
    st = st_gains - st_losses - st_cf
    lt = lt_gains - lt_losses - lt_cf
    investment_income = interest + dividends
    # Schedule B first applies ST losses against interest/dividends (line 20),
    # THEN against LT gains (line 22). This differs from federal Schedule D.
    st_income_offset = min(max(-st, 0.), investment_income, 2000.)
    st += st_income_offset
    investment_income -= st_income_offset
    if st < 0 < lt:
        matched = min(-st, lt)
        st += matched; lt -= matched
    elif lt < 0 < st:
        matched = min(-lt, st)
        lt += matched; st -= matched
    lt_income_offset = min(max(-lt, 0.), investment_income, 2000. - st_income_offset)
    lt += lt_income_offset
    investment_income -= lt_income_offset
    five_percent_base = ordinary_income - exemption + investment_income + max(lt, 0.)
    short_base = max(st, 0.)
    surtax_base = max(0., five_percent_base + short_base - SURTAX_THRESHOLDS[year])
    return {'total_tax': .05 * five_percent_base + .085 * short_base + .04 * surtax_base,
            'five_percent_tax': .05 * five_percent_base, 'short_term_tax': .085 * short_base,
            'surtax': .04 * surtax_base, 'st_cf': max(-st, 0.), 'lt_cf': max(-lt, 0.),
            'loss_offset_interest_dividends': st_income_offset + lt_income_offset,
            'exemption': exemption, 'surtax_threshold': SURTAX_THRESHOLDS[year],
            'tax_year': year, 'scope': scope}
