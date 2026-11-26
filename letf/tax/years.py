"""Versioned federal rate parameters. No extrapolation to unknown tax years.

Rev. Procs. 2020-45, 2021-45, 2022-38, 2023-34, 2024-40 and 2025-32 (the last updates 2025's
standard deductions after P.L. 119-21). Retrieved 2026-11-23.
These parameters do not implement credits, AMT or the entire Form 1040.
"""
from letf.accuracy import UnsupportedModelError

RATES = (.10, .12, .22, .24, .32, .35, .37)
ORDINARY = {
    2021: {
        'single': (9950, 40525, 86375, 164925, 209425, 523600),
        'married': (19900, 81050, 172750, 329850, 418850, 628300),
        'head_of_household': (14200, 54200, 86350, 164900, 209400, 523600),
        'married_filing_separately': (9950, 40525, 86375, 164925, 209425, 314150),
    },
    2022: {
        'single': (10275, 41775, 89075, 170050, 215950, 539900),
        'married': (20550, 83550, 178150, 340100, 431900, 647850),
        'head_of_household': (14650, 55900, 89050, 170050, 215950, 539900),
        'married_filing_separately': (10275, 41775, 89075, 170050, 215950, 323925),
    },
    2023: {
        'single': (11000, 44725, 95375, 182100, 231250, 578125),
        'married': (22000, 89450, 190750, 364200, 462500, 693750),
        'head_of_household': (15700, 59850, 95350, 182100, 231250, 578100),
        'married_filing_separately': (11000, 44725, 95375, 182100, 231250, 346875),
    },
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
    2021: {'single': (40400, 445850), 'married': (80800, 501600),
           'head_of_household': (54100, 473750), 'married_filing_separately': (40400, 250800)},
    2022: {'single': (41675, 459750), 'married': (83350, 517200),
           'head_of_household': (55800, 488500), 'married_filing_separately': (41675, 258600)},
    2023: {'single': (44625, 492300), 'married': (89250, 553850),
           'head_of_household': (59750, 523050), 'married_filing_separately': (44625, 276900)},
    2024: {'single': (47025, 518900), 'married': (94050, 583750),
           'head_of_household': (63000, 551350), 'married_filing_separately': (47025, 291850)},
    2025: {'single': (48350, 533400), 'married': (96700, 600050),
           'head_of_household': (64750, 566700), 'married_filing_separately': (48350, 300000)},
    2026: {'single': (49450, 545500), 'married': (98900, 613700),
           'head_of_household': (66200, 579600), 'married_filing_separately': (49450, 306850)},
}
DEDUCTIONS = {
    2021: {'single': 12550, 'married': 25100, 'head_of_household': 18800, 'married_filing_separately': 12550},
    2022: {'single': 12950, 'married': 25900, 'head_of_household': 19400, 'married_filing_separately': 12950},
    2023: {'single': 13850, 'married': 27700, 'head_of_household': 20800, 'married_filing_separately': 13850},
    2024: {'single': 14600, 'married': 29200, 'head_of_household': 21900, 'married_filing_separately': 14600},
    2025: {'single': 15750, 'married': 31500, 'head_of_household': 23625, 'married_filing_separately': 15750},
    2026: {'single': 16100, 'married': 32200, 'head_of_household': 24150, 'married_filing_separately': 16100},
}
# Parameters above are the basic standard deduction for an independent,
# nonblind taxpayer under 65. Additional aged/blind/dependent deductions and
# household credits need their own explicit inputs; these are not Form 1040.
HISTORICAL_AUTHORITIES = {
    2021: 'https://www.irs.gov/pub/irs-drop/rp-20-45.pdf',
    2022: 'https://www.irs.gov/pub/irs-drop/rp-21-45.pdf',
    2023: 'https://www.irs.gov/pub/irs-drop/rp-22-38.pdf',
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
