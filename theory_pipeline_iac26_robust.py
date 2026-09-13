#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAGNETIC HABITABILITY PIPELINE — CORRECTED VERSION

Purpose
-------
Read a CSV containing the selected planet names, retrieve the physical
parameters from the NASA Exoplanet Archive PSCompPars table, and evaluate
an HZ + stellar-wind + magnetopause + dynamo magnetic-shielding model.

Important methodological choices
---------------------------------
1. The input CSV is used ONLY to select planet names.
2. Physical parameters used in the calculation are taken from NASA
   Exoplanet Archive PSCompPars.
3. The conservative habitable zone is calculated consistently with the
   temperature-dependent Kopparapu et al. prescription (runaway-greenhouse
   inner edge and maximum-greenhouse outer edge).
4. The stellar mass-loss multiplier A is sampled log-uniformly from 0.3–10.
5. Dynamo parameters are sampled for every Monte-Carlo realization:
      r_c/R_p       ~ Uniform(0.35, 0.60)
      rho_c/rho_p   ~ Uniform(2, 4)
      F_F/F_Earth   ~ LogUniform(0.1, 10)
6. Available asymmetric NASA uncertainties are propagated with an
   asymmetric-normal approximation. If an uncertainty is unavailable,
   the reported central value is used deterministically.
7. Four magnetopause standoff configurations are evaluated:
      R_s/R_p = 5, 10, 15, 20
   These are sensitivity configurations, NOT habitability boundaries.
8. The output reports median, 16th/84th percentiles, and the Monte-Carlo
   fraction satisfying M_hab >= 1.
9. A target can remain in the 165-planet input sample even if some physical
   calculation cannot be performed because required NASA parameters are
   missing. Such cases are explicitly flagged rather than silently dropped.

The magnetic-habitability coefficient is

    M_hab = M_dyn / M_min

and M_hab >= 1 means that the sampled dynamo moment is sufficient for the
adopted magnetopause standoff requirement under the adopted stellar-wind
model. It is NOT, by itself, a direct biological habitability criterion.
"""

import argparse
import csv
import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.special import lambertw

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

warnings.filterwarnings("ignore")


# =====================================================================
# USER SETTINGS
# =====================================================================

INPUT_FILE = "final_dataset_iac26.csv"

MATCHED_FILE = "NASA_matched_planets.csv"
UNMATCHED_FILE = "NASA_unmatched_planets.csv"
RESULT_FILE = "magnetic_habitability_results.csv"
SUMMARY_FILE = "magnetic_habitability_summary.csv"

# Reproducibility.
RANDOM_SEED = 26026
N_MONTE_CARLO = 10_000

# Four sensitivity configurations. They are not habitability thresholds.
STANDOFF_FACTORS = np.array([5.0, 10.0, 15.0, 20.0])

# Magnetopause model parameters.
F0 = 1.5
KAPPA = 1.0

# Stellar-wind model.
M_SUN_LOSS = 1.26e9       # kg s^-1
ACTIVITY_MIN = 0.3
ACTIVITY_MAX = 10.0
MASS_LOSS_EXPONENT = 2.0

# Wind closure.
T_SW_FACTOR = 0.8
MU_MEAN = 0.5

# Dynamo model.
DYNAMO_C = 0.9
CORE_RADIUS_MIN = 0.35
CORE_RADIUS_MAX = 0.60
CORE_DENSITY_RATIO_MIN = 2.0
CORE_DENSITY_RATIO_MAX = 4.0
FLUX_RATIO_MIN = 0.1
FLUX_RATIO_MAX = 10.0
F_EARTH = 0.05             # W m^-2; adopted reference flux in this model

# Kopparapu validity range for this implementation.
KOPPARAPU_TMIN = 2600.0
KOPPARAPU_TMAX = 7200.0

# =====================================================================
# PHYSICAL CONSTANTS
# =====================================================================

G = 6.67430e-11
MU0 = 4.0 * np.pi * 1e-7
M_PROTON = 1.67262192369e-27
K_B = 1.380649e-23

AU = 1.495978707e11
M_SUN = 1.98847e30
R_SUN = 6.957e8
M_JUP = 1.89813e27
R_JUP = 7.1492e7


# =====================================================================
# BASIC HELPERS
# =====================================================================

def clean_numeric(value):
    """Convert a value to float; return NaN if conversion fails."""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def normalize_text(value):
    """Normalize Unicode, punctuation variants, case, and whitespace."""
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower()

    for old, new in [("–", "-"), ("—", "-"), ("−", "-")]:
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name(value):
    """Aggressive normalization for planet-name matching."""
    value = normalize_text(value)
    return re.sub(r"[^a-z0-9]", "", value)


def generate_name_variants(name):
    """Generate normalized representations of a planet name."""
    if pd.isna(name):
        return set()

    original = normalize_text(name)
    if not original:
        return set()

    variants = {
        original,
        original.replace(" ", ""),
        original.replace("-", ""),
        re.sub(r"[\s-]", "", original),
        re.sub(r"[^a-z0-9]", "", original),
    }

    # Host + planet letter variants, e.g. "Kepler-22 b" -> "Kepler-22b".
    match = re.match(r"(.+?)[\s-]+([bcdefghijk])$", original)
    if match:
        host, letter = match.groups()
        variants.add(host + letter)
        variants.add(host.replace("-", "") + letter)

    return {normalize_name(v) for v in variants if v}


def detect_delimiter(filename):
    """Detect the delimiter used by the input CSV."""
    with open(filename, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(10000)

    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        return ";"


def identify_planet_column(columns):
    """Find the planet-name column robustly, including case differences."""
    candidates = [
        "planet-name", "planet_name", "planet", "pl_name",
        "Planet", "Planet Name", "Planet_Name", "name", "Name"
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_columns = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_columns:
            return lower_columns[key]

    raise ValueError(
        "Could not identify the planet-name column. Available columns: "
        + ", ".join(map(str, columns))
    )


# =====================================================================
# NASA DATA
# =====================================================================

def get_archive_data():
    """Retrieve the required PSCompPars columns from NASA."""
    archive_columns = """
        pl_name, hostname, pl_letter,
        pl_massj, pl_massjerr1, pl_massjerr2,
        pl_radj, pl_radjerr1, pl_radjerr2,
        pl_orbsmax, pl_orbsmaxerr1, pl_orbsmaxerr2,
        pl_orbper, pl_orbpererr1, pl_orbpererr2,
        st_mass, st_masserr1, st_masserr2,
        st_rad, st_raderr1, st_raderr2,
        st_teff, st_tefferr1, st_tefferr2,
        st_lum, st_lumerr1, st_lumerr2,
        st_spectype
    """

    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select=archive_columns
    )
    return table.to_pandas()


def build_nasa_lookup(archive_df):
    """Build a normalized planet-name lookup into the NASA dataframe."""
    lookup = {}

    for index, row in archive_df.iterrows():
        variants = set(generate_name_variants(row.get("pl_name", "")))

        hostname = row.get("hostname", "")
        pl_letter = row.get("pl_letter", "")

        if pd.notna(hostname) and pd.notna(pl_letter):
            hostname = str(hostname).strip()
            pl_letter = str(pl_letter).strip()
            for combination in (
                hostname + pl_letter,
                hostname + " " + pl_letter,
                hostname + "-" + pl_letter,
            ):
                variants.update(generate_name_variants(combination))

        for variant in variants:
            if variant and variant not in lookup:
                lookup[variant] = index

    return lookup


def match_planets(input_planets, archive_df):
    """Match input targets to PSCompPars, with a NASA direct fallback."""
    lookup = build_nasa_lookup(archive_df)
    matched_rows = []
    unmatched = []

    for planet in input_planets:
        variants = generate_name_variants(planet)
        found_index = next((lookup[v] for v in variants if v in lookup), None)

        if found_index is not None:
            row = archive_df.loc[found_index].copy()
            row["input_planet"] = planet
            row["match_type"] = "normalized_exact"
            matched_rows.append(row)
            continue

        # Direct NASA lookup for targets not found in the downloaded table.
        try:
            target_table = NasaExoplanetArchive.query_object(
                planet,
                table="pscomppars",
                regularize=True
            )
            target_df = target_table.to_pandas()
        except Exception as exc:
            print(f"[NASA FALLBACK ERROR] {planet}: {exc}")
            unmatched.append(planet)
            continue

        if target_df.empty:
            unmatched.append(planet)
            continue

        input_variants = variants
        target_index = None

        for i, target_row in target_df.iterrows():
            target_variants = generate_name_variants(target_row.get("pl_name", ""))
            if input_variants & target_variants:
                target_index = i
                break

        if target_index is None:
            # query_object() already resolved the object; use its first row.
            target_index = target_df.index[0]

        row = target_df.loc[target_index].copy()
        row["input_planet"] = planet
        row["match_type"] = "NASA_query_object"
        matched_rows.append(row)

    if matched_rows:
        matched_df = pd.DataFrame(matched_rows).drop_duplicates("input_planet")
    else:
        matched_df = pd.DataFrame()

    unmatched_df = pd.DataFrame({"input_planet": unmatched})
    return matched_df, unmatched_df


# =====================================================================
# SPECTRAL CLASSIFICATION
# =====================================================================

def classify_spectral_type(spectral_type):
    """
    Return (spectral_group, main_sequence_flag).

    The classification is deliberately conservative. A missing luminosity
    class is not assumed to be main sequence; it is marked unknown.
    """
    text = normalize_text(spectral_type).upper()
    group_match = re.search(r"\b([FGKM])\s*\d(?:\.\d+)?", text)

    if group_match is None:
        # A few archive strings can be just "G", "K", or "M".
        simple = re.match(r"^\s*([FGKM])(?:\s|$)", text)
        group = simple.group(1) if simple else "UNKNOWN"
    else:
        group = group_match.group(1)

    if group not in {"G", "K", "M"}:
        return group, False

    # Luminosity class V or IV-V / V-IV is accepted as main sequence.
    if re.search(r"(?:^|[\s/()-])V(?:$|[\s/()-])", text) or re.search(r"V$", text):
        return group, True

    # Bare G/K/M classifications have insufficient information.
    if re.fullmatch(r"[GKM](?:\s*\d(?:\.\d+)?)?", text):
        return group, False

    return group, False


# =====================================================================
# KOPPARAPU HABITABLE ZONE
# =====================================================================

def kopparapu_hz(T_eff_K, L_star):
    """
    Conservative HZ using Kopparapu polynomial limits.

    Inner edge: runaway greenhouse.
    Outer edge: maximum greenhouse.

    Returns (inner_AU, outer_AU, valid_temperature_range).
    """
    if not np.isfinite(T_eff_K) or not np.isfinite(L_star) or L_star <= 0:
        return np.nan, np.nan, False

    valid_range = KOPPARAPU_TMIN <= T_eff_K <= KOPPARAPU_TMAX
    if not valid_range:
        return np.nan, np.nan, False

    T_prime = T_eff_K - 5780.0

    S_inner = (
        1.107
        + 1.332e-4 * T_prime
        - 1.580e-8 * T_prime**2
        + 8.308e-12 * T_prime**3
        - 1.931e-15 * T_prime**4
    )

    S_outer = (
        0.356
        + 6.171e-5 * T_prime
        + 1.698e-9 * T_prime**2
        - 3.198e-12 * T_prime**3
        + 5.575e-16 * T_prime**4
    )

    if S_inner <= 0 or S_outer <= 0:
        return np.nan, np.nan, False

    return np.sqrt(L_star / S_inner), np.sqrt(L_star / S_outer), True


# =====================================================================
# UNCERTAINTY SAMPLING
# =====================================================================

def sample_asymmetric_normal(mu, err_plus, err_minus, size, rng, positive=True):
    """
    Draw an asymmetric-normal approximation to a NASA measurement.

    Positive-side draws use err_plus; negative-side draws use |err_minus|.
    If no usable uncertainty exists, return the central value for every draw.
    Positive physical quantities are resampled if a draw is non-positive.
    """
    mu = clean_numeric(mu)
    ep = clean_numeric(err_plus)
    em = clean_numeric(err_minus)

    if not np.isfinite(mu):
        return np.full(size, np.nan)

    ep = abs(ep) if np.isfinite(ep) else np.nan
    em = abs(em) if np.isfinite(em) else np.nan

    if not np.isfinite(ep) and not np.isfinite(em):
        return np.full(size, mu, dtype=float)

    if not np.isfinite(ep):
        ep = em
    if not np.isfinite(em):
        em = ep

    z = rng.normal(0.0, 1.0, size)
    sigma = np.where(z >= 0.0, ep, em)
    samples = mu + z * sigma

    if positive:
        bad = ~np.isfinite(samples) | (samples <= 0.0)
        attempts = 0
        while np.any(bad) and attempts < 20:
            n_bad = int(np.sum(bad))
            z_new = rng.normal(0.0, 1.0, n_bad)
            sigma_new = np.where(z_new >= 0.0, ep, em)
            samples[bad] = mu + z_new * sigma_new
            bad = ~np.isfinite(samples) | (samples <= 0.0)
            attempts += 1

        # If the reported uncertainty is so large that rejection still
        # fails, use the positive central value rather than inventing data.
        samples[bad] = mu

    return samples


def sample_log_uniform(low, high, size, rng):
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size)


# =====================================================================
# STELLAR WIND MODEL
# =====================================================================

def stellar_wind_temperature(T_eff):
    if not np.isfinite(T_eff) or T_eff <= 0:
        return np.nan
    return T_SW_FACTOR * T_eff


def sound_speed(T_sw):
    if not np.isfinite(T_sw) or T_sw <= 0:
        return np.nan
    return np.sqrt(K_B * T_sw / (MU_MEAN * M_PROTON))


def parker_velocity_vectorized(r, R_star, M_star, T_sw):
    """
    Vectorized isothermal Parker-wind solution.

    Using the Lambert-W analytic form avoids millions of scalar root solves
    during the Monte-Carlo calculation.
    """
    r = np.asarray(r, dtype=float)
    R_star = np.asarray(R_star, dtype=float)
    M_star = np.asarray(M_star, dtype=float)
    T_sw = np.asarray(T_sw, dtype=float)

    a = sound_speed_array(T_sw)
    rc = G * M_star / (2.0 * a**2)
    x = r / rc
    rhs = 4.0 * np.log(x) + 4.0 / x - 3.0

    result = np.full_like(x, np.nan, dtype=float)
    good = (
        np.isfinite(r) & np.isfinite(R_star) & np.isfinite(M_star)
        & np.isfinite(T_sw) & np.isfinite(a) & np.isfinite(rc)
        & (r > R_star) & (R_star > 0) & (M_star > 0)
        & (T_sw > 0) & (a > 0) & (rc > 0) & (x > 0)
    )

    if not np.any(good):
        return result

    xg = x[good]
    rhsg = rhs[good]
    branch = np.where(xg < 1.0, 0, -1)
    W = lambertw(-np.exp(-rhsg), branch).real
    y = -W

    valid = np.isfinite(y) & (y > 0)
    values = np.full_like(y, np.nan)
    values[valid] = a[good][valid] * np.sqrt(y[valid])
    result[good] = values
    return result


def sound_speed_array(T_sw):
    T_sw = np.asarray(T_sw, dtype=float)
    return np.sqrt(K_B * T_sw / (MU_MEAN * M_PROTON))


def stellar_mass_loss_vectorized(M_star_solar, activity_factor):
    M_star_solar = np.asarray(M_star_solar, dtype=float)
    activity_factor = np.asarray(activity_factor, dtype=float)
    return (
        M_SUN_LOSS
        * M_star_solar**MASS_LOSS_EXPONENT
        * activity_factor
    )


# =====================================================================
# MAGNETOPAUSE + DYNAMO
# =====================================================================

def minimum_magnetic_moment_vectorized(R_s, Mdot, v_eff, r_orb):
    """Required dipole moment for the adopted standoff equation."""
    return (
        R_s**3
        * np.sqrt(
            (2.0 * np.pi * Mdot * KAPPA * v_eff)
            / (MU0 * F0**2 * r_orb**2)
        )
    )


def dynamo_magnetic_moment_vectorized(R_p, M_p, rng, size):
    """
    Sample the dynamo parameter space and return M_dyn plus sampled inputs.
    """
    core_radius_fraction = rng.uniform(
        CORE_RADIUS_MIN, CORE_RADIUS_MAX, size
    )
    core_density_ratio = rng.uniform(
        CORE_DENSITY_RATIO_MIN, CORE_DENSITY_RATIO_MAX, size
    )
    flux_ratio = sample_log_uniform(
        FLUX_RATIO_MIN, FLUX_RATIO_MAX, size, rng
    )

    rho_p = M_p / ((4.0 / 3.0) * np.pi * R_p**3)
    rho_c = core_density_ratio * rho_p
    F_F = flux_ratio * F_EARTH

    r_c = core_radius_fraction * R_p

    B_cmb = (
        DYNAMO_C
        * np.sqrt(MU0)
        * rho_c**(1.0 / 6.0)
        * F_F**(1.0 / 3.0)
    )

    B_p = B_cmb * (r_c / R_p)**3

    M_dyn = 4.0 * np.pi * B_p * R_p**3 / MU0

    return (
        M_dyn,
        core_radius_fraction,
        core_density_ratio,
        flux_ratio,
        rho_p,
    )


# =====================================================================
# ONE PLANET
# =====================================================================

def run_planet_monte_carlo(row, rng):
    """Run the complete Monte-Carlo calculation for one target."""

    n = N_MONTE_CARLO

    # -----------------------------------------------------------------
    # Observational parameters from NASA PSCompPars.
    # -----------------------------------------------------------------
    M_p_jup = sample_asymmetric_normal(
        row.get("pl_massj"), row.get("pl_massjerr1"), row.get("pl_massjerr2"),
        n, rng, positive=True
    )
    R_p_jup = sample_asymmetric_normal(
        row.get("pl_radj"), row.get("pl_radjerr1"), row.get("pl_radjerr2"),
        n, rng, positive=True
    )
    a_AU = sample_asymmetric_normal(
        row.get("pl_orbsmax"), row.get("pl_orbsmaxerr1"), row.get("pl_orbsmaxerr2"),
        n, rng, positive=True
    )
    P_days = sample_asymmetric_normal(
        row.get("pl_orbper"), row.get("pl_orbpererr1"), row.get("pl_orbpererr2"),
        n, rng, positive=True
    )
    M_star_solar = sample_asymmetric_normal(
        row.get("st_mass"), row.get("st_masserr1"), row.get("st_masserr2"),
        n, rng, positive=True
    )
    R_star_solar = sample_asymmetric_normal(
        row.get("st_rad"), row.get("st_raderr1"), row.get("st_raderr2"),
        n, rng, positive=True
    )
    T_eff = sample_asymmetric_normal(
        row.get("st_teff"), row.get("st_tefferr1"), row.get("st_tefferr2"),
        n, rng, positive=True
    )
    log_L_star = sample_asymmetric_normal(
        row.get("st_lum"), row.get("st_lumerr1"), row.get("st_lumerr2"),
        n, rng, positive=False
    )

    # -----------------------------------------------------------------
    # Unit conversion.
    # -----------------------------------------------------------------
    M_p = M_p_jup * M_JUP
    R_p = R_p_jup * R_JUP
    a = a_AU * AU
    M_star = M_star_solar * M_SUN
    R_star = R_star_solar * R_SUN
    L_star = 10.0 ** log_L_star

    # -----------------------------------------------------------------
    # HZ.
    # -----------------------------------------------------------------
    hz_inner = np.full(n, np.nan)
    hz_outer = np.full(n, np.nan)
    hz_valid_temp = (
        np.isfinite(T_eff)
        & (T_eff >= KOPPARAPU_TMIN)
        & (T_eff <= KOPPARAPU_TMAX)
        & np.isfinite(L_star)
        & (L_star > 0)
    )

    Tp = T_eff[hz_valid_temp] - 5780.0
    S_inner = (
        1.107 + 1.332e-4 * Tp - 1.580e-8 * Tp**2
        + 8.308e-12 * Tp**3 - 1.931e-15 * Tp**4
    )
    S_outer = (
        0.356 + 6.171e-5 * Tp + 1.698e-9 * Tp**2
        - 3.198e-12 * Tp**3 + 5.575e-16 * Tp**4
    )

    hz_inner[hz_valid_temp] = np.sqrt(L_star[hz_valid_temp] / S_inner)
    hz_outer[hz_valid_temp] = np.sqrt(L_star[hz_valid_temp] / S_outer)
    in_hz = (
        np.isfinite(a_AU) & np.isfinite(hz_inner) & np.isfinite(hz_outer)
        & (a_AU >= hz_inner) & (a_AU <= hz_outer)
    )

    # -----------------------------------------------------------------
    # Stellar wind.
    # -----------------------------------------------------------------
    T_sw = T_SW_FACTOR * T_eff
    activity_factor = sample_log_uniform(
        ACTIVITY_MIN, ACTIVITY_MAX, n, rng
    )
    Mdot = stellar_mass_loss_vectorized(M_star_solar, activity_factor)
    v_eff = parker_velocity_vectorized(a, R_star, M_star, T_sw)

    wind_good = (
        np.isfinite(Mdot) & np.isfinite(v_eff) & np.isfinite(a)
        & (Mdot > 0) & (v_eff > 0) & (a > 0)
    )

    # Number density of protons. KAPPA is retained as the adopted density
    # scaling in the manuscript's pressure model.
    eta = np.full(n, np.nan)
    eta[wind_good] = (
        Mdot[wind_good] * KAPPA
        / (4.0 * np.pi * a[wind_good]**2 * M_PROTON * v_eff[wind_good])
    )

    P_dynamic = np.full(n, np.nan)
    P_dynamic[wind_good] = eta[wind_good] * M_PROTON * v_eff[wind_good]**2

    P_thermal = np.full(n, np.nan)
    thermal_good = wind_good & np.isfinite(T_sw)
    P_thermal[thermal_good] = eta[thermal_good] * K_B * T_sw[thermal_good]

    # -----------------------------------------------------------------
    # Dynamo.
    # -----------------------------------------------------------------
    dynamo_good = (
        np.isfinite(M_p) & np.isfinite(R_p)
        & (M_p > 0) & (R_p > 0)
    )

    M_dyn = np.full(n, np.nan)
    core_radius_fraction = np.full(n, np.nan)
    core_density_ratio = np.full(n, np.nan)
    flux_ratio = np.full(n, np.nan)
    rho_p = np.full(n, np.nan)

    if dynamo_good.any():
        # The sampled dynamo parameters are generated for all realizations,
        # then invalid realizations are masked below.
        Md, crf, cdr, fr, rp = dynamo_magnetic_moment_vectorized(
            R_p, M_p, rng, n
        )
        M_dyn = Md
        core_radius_fraction = crf
        core_density_ratio = cdr
        flux_ratio = fr
        rho_p = rp

    # A planet needs both a valid wind solution and a valid dynamo input.
    valid = wind_good & dynamo_good & np.isfinite(M_dyn)

    # -----------------------------------------------------------------
    # Output one summary row per standoff configuration.
    # -----------------------------------------------------------------
    output_rows = []

    # Central-value descriptors are useful in the 660-row result table.
    def central_value(value):
        return clean_numeric(value)

    planet = row.get("input_planet", "")
    nasa_name = row.get("pl_name", "")
    hostname = row.get("hostname", "")
    spectral_type = row.get("st_spectype", "")
    spectral_group, is_gkm_ms = classify_spectral_type(spectral_type)

    central_M_p = central_value(row.get("pl_massj"))
    central_R_p = central_value(row.get("pl_radj"))
    central_a = central_value(row.get("pl_orbsmax"))
    central_P = central_value(row.get("pl_orbper"))
    central_Mstar = central_value(row.get("st_mass"))
    central_Rstar = central_value(row.get("st_rad"))
    central_Teff = central_value(row.get("st_teff"))
    central_logL = central_value(row.get("st_lum"))
    central_L = 10.0**central_logL if np.isfinite(central_logL) else np.nan
    central_hzi, central_hzo, central_hz_temp_valid = kopparapu_hz(
        central_Teff, central_L
    )
    central_in_hz = (
        np.isfinite(central_a)
        and np.isfinite(central_hzi)
        and np.isfinite(central_hzo)
        and central_hzi <= central_a <= central_hzo
    )

    # Median sampled wind quantities for reporting.
    def median_if_valid(arr):
        vals = arr[np.isfinite(arr)]
        return float(np.median(vals)) if vals.size else np.nan

    median_Tsw = median_if_valid(T_sw)
    median_Mdot = median_if_valid(Mdot)
    median_v = median_if_valid(v_eff)
    median_eta = median_if_valid(eta)
    median_Pdyn = median_if_valid(P_dynamic)
    median_Ptherm = median_if_valid(P_thermal)

    for standoff_factor in STANDOFF_FACTORS:
        R_s = standoff_factor * R_p
        M_min = np.full(n, np.nan)

        good = valid & np.isfinite(R_s)
        M_min[good] = minimum_magnetic_moment_vectorized(
            R_s[good], Mdot[good], v_eff[good], a[good]
        )

        M_hab = np.full(n, np.nan)
        mh_good = good & np.isfinite(M_min) & (M_min > 0)
        M_hab[mh_good] = M_dyn[mh_good] / M_min[mh_good]

        finite = M_hab[np.isfinite(M_hab)]
        if finite.size:
            median_mhab = float(np.median(finite))
            p16_mhab, p84_mhab = np.percentile(finite, [16, 84])
            probability = float(np.mean(finite >= 1.0))
            valid_mc = int(finite.size)
        else:
            median_mhab = np.nan
            p16_mhab = np.nan
            p84_mhab = np.nan
            probability = np.nan
            valid_mc = 0

        # Median moments are calculated from the same realization ensemble.
        finite_dyn = M_dyn[np.isfinite(M_dyn)]
        finite_min = M_min[np.isfinite(M_min)]

        output_rows.append({
            "planet_name": planet,
            "nasa_pl_name": nasa_name,
            "hostname": hostname,
            "spectral_type": spectral_type,
            "spectral_group": spectral_group,
            "gkm_main_sequence": is_gkm_ms,
            "match_type": row.get("match_type", ""),

            "planet_mass_Mjup": central_M_p,
            "planet_radius_Rjup": central_R_p,
            "semi_major_axis_AU": central_a,
            "orbital_period_days": central_P,
            "stellar_mass_Msun": central_Mstar,
            "stellar_radius_Rsun": central_Rstar,
            "stellar_teff_K": central_Teff,
            "stellar_luminosity_log10_Lsun": central_logL,
            "stellar_luminosity_Lsun": central_L,

            "hz_inner_AU": central_hzi,
            "hz_outer_AU": central_hzo,
            "hz_temperature_valid": central_hz_temp_valid,
            "in_habitable_zone": central_in_hz,

            "monte_carlo_realizations": N_MONTE_CARLO,
            "valid_monte_carlo_realizations": valid_mc,

            "stellar_wind_temperature_median_K": median_Tsw,
            "stellar_mass_loss_median_kg_s": median_Mdot,
            "stellar_wind_velocity_median_m_s": median_v,
            "stellar_wind_density_median_m-3": median_eta,
            "stellar_wind_dynamic_pressure_median_Pa": median_Pdyn,
            "stellar_wind_thermal_pressure_median_Pa": median_Ptherm,

            "standoff_factor_Rs_Rp": float(standoff_factor),
            "standoff_distance_m_median": (
                float(np.nanmedian(R_s)) if np.isfinite(R_s).any() else np.nan
            ),
            "minimum_magnetic_moment_median_A_m2": (
                float(np.median(finite_min)) if finite_min.size else np.nan
            ),
            "dynamo_magnetic_moment_median_A_m2": (
                float(np.median(finite_dyn)) if finite_dyn.size else np.nan
            ),

            "magnetic_habitability_median": median_mhab,
            "magnetic_habitability_p16": float(p16_mhab),
            "magnetic_habitability_p84": float(p84_mhab),
            "P_Mhab_ge_1": probability,
            "magnetically_sufficient": (
                bool(probability >= 0.5) if np.isfinite(probability) else False
            ),
        })

    return output_rows


# =====================================================================
# MAIN
# =====================================================================

def main():
    global N_MONTE_CARLO

    parser = argparse.ArgumentParser(description="Magnetic habitability pipeline")
    parser.add_argument("--input", default=INPUT_FILE, help="Input CSV")
    parser.add_argument("--output", default=RESULT_FILE, help="Detailed result CSV")
    parser.add_argument("--summary", default=SUMMARY_FILE, help="Population summary CSV")
    parser.add_argument("--matched", default=MATCHED_FILE, help="NASA matched CSV")
    parser.add_argument("--unmatched", default=UNMATCHED_FILE, help="NASA unmatched CSV")
    parser.add_argument("--n-mc", type=int, default=N_MONTE_CARLO,
                        help="Monte-Carlo realizations per target")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed")
    args = parser.parse_args()

    N_MONTE_CARLO = int(args.n_mc)
    if N_MONTE_CARLO <= 0:
        raise ValueError("--n-mc must be positive.")

    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("MAGNETIC HABITABILITY PIPELINE — CORRECTED")
    print("=" * 78)
    print(f"Input: {args.input}")
    print(f"Monte-Carlo realizations per planet: {N_MONTE_CARLO:,}")
    print(f"Random seed: {args.seed}")

    # -----------------------------------------------------------------
    # Input.
    # -----------------------------------------------------------------
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    delimiter = detect_delimiter(args.input)
    input_df = pd.read_csv(args.input, sep=delimiter, encoding="utf-8-sig")
    input_df.columns = input_df.columns.astype(str).str.strip()

    planet_column = identify_planet_column(input_df.columns)
    input_planets = (
        input_df[planet_column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    input_planets = [p for p in input_planets if p]
    input_planets = list(dict.fromkeys(input_planets))

    print(f"Detected delimiter: {repr(delimiter)}")
    print(f"Planet-name column: {planet_column}")
    print(f"Selected unique planets: {len(input_planets)}")

    # -----------------------------------------------------------------
    # NASA.
    # -----------------------------------------------------------------
    print("\nDownloading NASA PSCompPars data...")
    archive_df = get_archive_data()
    print(f"NASA returned {len(archive_df):,} rows.")

    matched_df, unmatched_df = match_planets(input_planets, archive_df)

    matched_df.to_csv(args.matched, index=False, float_format="%.15g", na_rep="")
    unmatched_df.to_csv(args.unmatched, index=False, na_rep="")

    print(f"NASA matched: {len(matched_df)}")
    print(f"NASA unmatched: {len(unmatched_df)}")

    if matched_df.empty:
        raise RuntimeError("No selected planets could be matched to NASA Exoplanet Archive.")

    # Convert NASA numeric columns once.
    numeric_columns = [
        "pl_massj", "pl_massjerr1", "pl_massjerr2",
        "pl_radj", "pl_radjerr1", "pl_radjerr2",
        "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2",
        "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
        "st_mass", "st_masserr1", "st_masserr2",
        "st_rad", "st_raderr1", "st_raderr2",
        "st_teff", "st_tefferr1", "st_tefferr2",
        "st_lum", "st_lumerr1", "st_lumerr2",
    ]
    for col in numeric_columns:
        if col in matched_df.columns:
            matched_df[col] = pd.to_numeric(matched_df[col], errors="coerce")

    # -----------------------------------------------------------------
    # Scientific calculation.
    # -----------------------------------------------------------------
    all_results = []
    for i, (_, row) in enumerate(matched_df.iterrows(), start=1):
        planet = row.get("input_planet", "")
        print(f"[{i:3d}/{len(matched_df):3d}] {planet}")
        try:
            all_results.extend(run_planet_monte_carlo(row, rng))
        except Exception as exc:
            print(f"    ERROR: {exc}")
            # Preserve all four target/configuration rows even when the
            # scientific calculation fails, so target accounting remains
            # transparent.
            spectral_group, is_ms = classify_spectral_type(row.get("st_spectype", ""))
            for factor in STANDOFF_FACTORS:
                all_results.append({
                    "planet_name": planet,
                    "nasa_pl_name": row.get("pl_name", ""),
                    "hostname": row.get("hostname", ""),
                    "spectral_type": row.get("st_spectype", ""),
                    "spectral_group": spectral_group,
                    "gkm_main_sequence": is_ms,
                    "match_type": row.get("match_type", ""),
                    "standoff_factor_Rs_Rp": float(factor),
                    "monte_carlo_realizations": N_MONTE_CARLO,
                    "valid_monte_carlo_realizations": 0,
                    "calculation_error": str(exc),
                    "magnetically_sufficient": False,
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output, index=False, float_format="%.15g", na_rep="")

    # -----------------------------------------------------------------
    # Population summary.
    # -----------------------------------------------------------------
    summary_rows = []
    for factor in STANDOFF_FACTORS:
        sub = results_df[results_df["standoff_factor_Rs_Rp"] == factor].copy()
        unique_targets = sub["planet_name"].nunique()
        valid_targets = sub.loc[
            sub["valid_monte_carlo_realizations"].fillna(0) > 0,
            "planet_name"
        ].nunique()

        mh = pd.to_numeric(sub.get("magnetic_habitability_median"), errors="coerce")
        prob = pd.to_numeric(sub.get("P_Mhab_ge_1"), errors="coerce")
        hz = sub.get("in_habitable_zone", pd.Series(dtype=bool))

        hz_mask = hz.fillna(False).astype(bool) if len(hz) else pd.Series(dtype=bool)
        hz_sub = sub.loc[hz_mask] if len(hz_mask) else sub.iloc[0:0]
        hz_prob = pd.to_numeric(hz_sub.get("P_Mhab_ge_1"), errors="coerce")

        summary_rows.append({
            "standoff_factor_Rs_Rp": float(factor),
            "input_targets_represented": unique_targets,
            "targets_with_valid_monte_carlo": valid_targets,
            "median_Mhab_across_targets": (
                float(mh.median()) if mh.notna().any() else np.nan
            ),
            "median_P_Mhab_ge_1_across_targets": (
                float(prob.median()) if prob.notna().any() else np.nan
            ),
            "HZ_targets": int(hz_sub["planet_name"].nunique()) if not hz_sub.empty else 0,
            "HZ_median_P_Mhab_ge_1": (
                float(hz_prob.median()) if hz_prob.notna().any() else np.nan
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary, index=False, float_format="%.15g", na_rep="")

    # -----------------------------------------------------------------
    # Final accounting.
    # -----------------------------------------------------------------
    valid_planets = results_df.loc[
        results_df["valid_monte_carlo_realizations"].fillna(0) > 0,
        "planet_name"
    ].nunique()
    hz_planets = results_df.loc[
        results_df.get("in_habitable_zone", False).fillna(False).astype(bool),
        "planet_name"
    ].nunique() if "in_habitable_zone" in results_df else 0

    gkm_ms = results_df.loc[
        results_df.get("gkm_main_sequence", False).fillna(False).astype(bool),
        "planet_name"
    ].nunique() if "gkm_main_sequence" in results_df else 0

    print("\n" + "=" * 78)
    print("CALCULATION COMPLETE")
    print("=" * 78)
    print(f"Input targets:                         {len(input_planets)}")
    print(f"NASA matched targets:                  {len(matched_df)}")
    print(f"NASA unmatched targets:                {len(unmatched_df)}")
    print(f"Result rows:                            {len(results_df)}")
    print(f"Expected rows (matched × 4):            {len(matched_df) * len(STANDOFF_FACTORS)}")
    print(f"Targets with valid MC calculations:     {valid_planets}")
    print(f"Targets in HZ (central NASA values):    {hz_planets}")
    print(f"G/K/M main-sequence targets identified: {gkm_ms}")
    print("\nOutput files:")
    print(f"  {args.matched}")
    print(f"  {args.unmatched}")
    print(f"  {args.output}")
    print(f"  {args.summary}")


if __name__ == "__main__":
    main()
