#! /usr/bin/env python3
#
# Physical constants for radar calculations
#
# Jan-2025, Pat Welch, pat@mousebrains.com

"""Physical constants for radar calculations."""

__all__ = ["C_VACUUM", "N_AIR_STANDARD", "C_AIR", "KNOTS_TO_MS"]

# Speed of light in vacuum (m/s)
C_VACUUM = 299_792_458.0

# Refractive index of air at standard conditions:
# 20°C, 50% relative humidity, 1013.25 hPa
# Based on Ciddor equation approximation
N_AIR_STANDARD = 1.000273

# Speed of light in air at standard conditions (m/s)
# ~299,710,639 m/s
C_AIR = C_VACUUM / N_AIR_STANDARD

# Unit conversions
KNOTS_TO_MS = 0.514444  # 1 knot = 0.514444 m/s
