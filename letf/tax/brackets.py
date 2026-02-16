STATE_TAX_BRACKETS = {
    'CA': {  # California
        'single': [
            (10412, 0.01), (24684, 0.02), (38959, 0.04), (54081, 0.06),
            (68350, 0.08), (349137, 0.093), (418961, 0.103),
            (698271, 0.113), (float('inf'), 0.133)
        ],
        'married': [
            (20824, 0.01), (49368, 0.02), (77918, 0.04), (108162, 0.06),
            (136700, 0.08), (698274, 0.093), (837922, 0.103),
            (1396542, 0.113), (float('inf'), 0.133)
        ],
        'std_deduction': {'single': 5363, 'married': 10726}
    },
    'NY': {  # New York
        'single': [
            (8500, 0.04), (11700, 0.045), (13900, 0.0525), (80650, 0.055),
            (215400, 0.06), (1077550, 0.0685), (5000000, 0.0965),
            (25000000, 0.103), (float('inf'), 0.109)
        ],
        'married': [
            (17150, 0.04), (23600, 0.045), (27900, 0.0525), (161550, 0.055),
            (323200, 0.06), (2155350, 0.0685), (5000000, 0.0965),
            (25000000, 0.103), (float('inf'), 0.109)
        ],
        'std_deduction': {'single': 8000, 'married': 16050}
    },
    'TX': {  # Texas (no state income tax)
        'single': [(float('inf'), 0.0)],
        'married': [(float('inf'), 0.0)],
        'std_deduction': {'single': 0, 'married': 0}
    },
    'FL': {  # Florida (no state income tax)
        'single': [(float('inf'), 0.0)],
        'married': [(float('inf'), 0.0)],
        'std_deduction': {'single': 0, 'married': 0}
    },
    'WA': {  # Washington (no general income tax, but 7% capital gains tax on gains >$250k)
        'single': [(250000, 0.0), (float('inf'), 0.07)],
        'married': [(250000, 0.0), (float('inf'), 0.07)],
        'std_deduction': {'single': 0, 'married': 0},
        'cap_gains_only': True  # WA tax applies to capital gains only, not ordinary income
    },
    'NV': {  # Nevada (no state income tax)
        'single': [(float('inf'), 0.0)],
        'married': [(float('inf'), 0.0)],
        'std_deduction': {'single': 0, 'married': 0}
    },
    'IL': {  # Illinois (flat tax)
        'single': [(float('inf'), 0.0495)],
        'married': [(float('inf'), 0.0495)],
        'std_deduction': {'single': 2425, 'married': 4850}
    },
    'MA': {  # Massachusetts
        'single': [(float('inf'), 0.05)],  # Flat 5%
        'married': [(float('inf'), 0.05)],
        'std_deduction': {'single': 0, 'married': 0}  # No standard deduction
    },
    'NJ': {  # New Jersey
        # NJ taxes capital gains as ordinary income (no preferential rate).
        # Uses personal exemptions instead of standard deduction.
        # Single exemption: $1,000. Married: $2,000.
        'single': [
            (20000, 0.014), (35000, 0.0175), (40000, 0.035),
            (75000, 0.05525), (500000, 0.0637), (1000000, 0.0897),
            (float('inf'), 0.1075)
        ],
        'married': [
            (20000, 0.014), (50000, 0.0175), (70000, 0.0245),
            (80000, 0.035), (150000, 0.05525), (500000, 0.0637),
            (1000000, 0.0897), (float('inf'), 0.1075)
        ],
        'std_deduction': {'single': 1000, 'married': 2000}
    }
}


# 2024 Tax Brackets by Filing Status
FEDERAL_TAX_BRACKETS_2024 = {
    'single': [
        (11600, 0.10), (47150, 0.12), (100525, 0.22), (191950, 0.24),
        (243725, 0.32), (609350, 0.35), (float('inf'), 0.37)
    ],
    'married': [
        (23200, 0.10), (94300, 0.12), (201050, 0.22), (383900, 0.24),
        (487450, 0.32), (731200, 0.35), (float('inf'), 0.37)
    ]
}

LTCG_BRACKETS_2024 = {
    'single': [
        (47025, 0.00), (518900, 0.15), (float('inf'), 0.20)
    ],
    'married': [
        (94050, 0.00), (583750, 0.15), (float('inf'), 0.20)
    ]
}

STANDARD_DEDUCTION_2024 = {
    'single': 14600,
    'married': 29200
}

NIIT_THRESHOLD_2024 = {
    'single': 200000,
    'married': 250000
}


# Keep old constants for backward compatibility
TAX_BRACKETS_2024 = FEDERAL_TAX_BRACKETS_2024  # Was with ['single']
LTCG_BRACKETS_2024_SINGLE = LTCG_BRACKETS_2024  # Rename if needed, remove _SINGLE
CA_TAX_BRACKETS = STATE_TAX_BRACKETS['CA']  # Keep as is if no ['single']
CA_STANDARD_DEDUCTION = STATE_TAX_BRACKETS['CA']['std_deduction']  # Remove ['single'] if present
STANDARD_DEDUCTION_2024_SINGLE = STANDARD_DEDUCTION_2024  # Remove _SINGLE
NIIT_RATE = 0.038
