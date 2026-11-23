"""Versioned federal rate parameters. No extrapolation to unknown tax years.

Rev. Procs. 2023-34, 2024-40 and 2025-32 (the last updates 2025's
standard deductions after P.L. 119-21). Retrieved 2026-11-23.
These parameters do not implement credits, AMT or the entire Form 1040.
"""
from letf.accuracy import UnsupportedModelError

RATES = (.10, .12, .22, .24, .32, .35, .37)
ORDINARY = {
    2024: {
        'single': (11600, 47150, 100525, 191950, 243725, 609350),
        'married': (23200, 94300, 201050, 383900, 487450, 731200),
        'head_of_household': (16550, 63100, 100500, 191950, 243700, 609350),
        'married_filing_separately': (11600, 47150, 100525, 191950, 243725, 365600),
    },
    2025: {
        'single': (11925, 48475, 103350, 197300, 250525, 626350),
        'married': (23850, 96950, 206700, 394600, 501050, 751600),
        'head_of_household': (17000, 64850, 103350, 197300, 250500, 626350),
        'married_filing_separately': (11925, 48475, 103350, 197300, 250525, 375800),
    },
    2026: {
        'single': (12400, 50400, 105700, 201775, 256225, 640600),
        'married': (24800, 100800, 211400, 403550, 512450, 768700),
        'head_of_household': (17700, 67450, 105700, 201750, 256200, 640600),
        'married_filing_separately': (12400, 50400, 105700, 201775, 256225, 384350),
    },
}
PREFERENTIAL = {
    2024: {'single': (47025, 518900), 'married': (94050, 583750),
           'head_of_household': (63000, 551350), 'married_filing_separately': (47025, 291850)},
    2025: {'single': (48350, 533400), 'married': (96700, 600050),
           'head_of_household': (64750, 566700), 'married_filing_separately': (48350, 300000)},
    2026: {'single': (49450, 545500), 'married': (98900, 613700),
           'head_of_household': (66200, 579600), 'married_filing_separately': (49450, 306850)},
}
DEDUCTIONS = {
    2024: {'single': 14600, 'married': 29200, 'head_of_household': 21900, 'married_filing_separately': 14600},
    2025: {'single': 15750, 'married': 31500, 'head_of_household': 23625, 'married_filing_separately': 15750},
    2026: {'single': 16100, 'married': 32200, 'head_of_household': 24150, 'married_filing_separately': 16100},
}
# IRC 1411(b) thresholds are NOT indexed for inflation.
NIIT_THRESHOLDS = {'single': 200000, 'married': 250000,
                   'head_of_household': 200000, 'married_filing_separately': 125000}


def federal_parameters(year, filing_status):
    status = {'married_filing_jointly': 'married'}.get(filing_status, filing_status)
    if year not in ORDINARY or status not in ORDINARY[year]:
        raise UnsupportedModelError(f'No verified federal tax table for {year}/{filing_status}')
    ordinary = list(zip((*ORDINARY[year][status], float('inf')), RATES))
    preferential = list(zip((*PREFERENTIAL[year][status], float('inf')), (0., .15, .20)))
    return ordinary, preferential, DEDUCTIONS[year][status], NIIT_THRESHOLDS[status]
