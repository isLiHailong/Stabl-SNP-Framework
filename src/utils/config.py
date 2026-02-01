DEFAULT_SEED = 42

NUMERIC_STABILITY = {
    "ridge": 1e-3,
    "eps": 1e-10,
    "jitter": 1e-6,
}

STABILITY_SELECTION = {
    "pgrid_left": 1e-8,
    "pgrid_right": 1e-4,
    "pgrid_size": 401,
    "theta_left": 0.1,
    "theta_right": 0.95,
    "theta_size": 86,
}


HWE_SETTINGS = {
    "missing_values": (-1, 9, 99),
    "p_threshold": 1e-4,
    "method": "exact",
}