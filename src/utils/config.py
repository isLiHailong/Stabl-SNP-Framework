DEFAULTSEED = 42

NUMERICSTABILITY = {
    "ridge": 1e-3,
    "eps": 1e-10,
    "jitter": 1e-6,
}

STABILITYSELECTION = {
    "pgridleft": 1e-8,
    "pgridright": 1e-4,
    "pgridsize": 401,
    "thetaleft": 0.1,
    "thetaright": 0.95,
    "thetasize": 86,
}

HWESETTINGS = {
    "missingvalues": (-1, 9, 99),
    "pthreshold": 1e-4,
    "method": "exact",
}
