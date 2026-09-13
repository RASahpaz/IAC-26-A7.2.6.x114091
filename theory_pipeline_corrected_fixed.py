#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:03:22 2026

@author: resulayberksahpaz
"""

# ============================================================
# MAGNETIC HABITABILITY ANALYSIS
#
# Input:
#     CSV file containing a "planet-name" column
#
# Example:
#
# planet-name;hostname;spectral_class;st_teff;luminosity;
# pl_orbsmax;hz_inner;hz_outer
#
# Workflow:
#
# CSV
#   |
#   +--> Read selected planet names
#   |
#   +--> NASA Exoplanet Archive PSCompPars
#   |
#   +--> Robust planet-name matching
#   |
#   +--> Stellar wind calculation
#   |
#   +--> Magnetopause calculation
#   |
#   +--> Dynamo magnetic moment
#   |
#   +--> Magnetic habitability coefficient
#   |
#   +--> CSV output
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import csv
import unicodedata
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import brentq

from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
    NasaExoplanetArchive
)


warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# INPUT FILE
# ------------------------------------------------------------

INPUT_FILE = "final_dataset_iac26.csv"


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

MATCHED_FILE = "NASA_matched_planets.csv"

UNMATCHED_FILE = "NASA_unmatched_planets.csv"

RESULT_FILE = "magnetic_habitability_results.csv"


# ============================================================
# MODEL SETTINGS
# ============================================================

# Magnetopause standoff distances in planetary radii.
#
# Example:
#
# Rs = 5 Rp
# Rs = 10 Rp
# Rs = 15 Rp
# Rs = 20 Rp
#

STANDOFF_FACTORS = [
    5.0,
    10.0,
    15.0,
    20.0
]


# Magnetopause model parameters

f0 = 1.5

kappa = 1.0


# ============================================================
# STELLAR WIND PARAMETERS
# ============================================================

# Solar mass-loss rate [kg/s]

M_SUN_LOSS = 1.26e9


# Activity multiplier.
#
# 1.0 = baseline solar-like activity.
#
# Change this according to the model adopted in your paper.

ACTIVITY_FACTOR = 1.0


# Stellar mass-loss scaling exponent.

MASS_LOSS_EXPONENT = 2.0


# ============================================================
# DYNAMO PARAMETERS
# ============================================================

dynamo_c = 0.9


# Core radius as fraction of planetary radius.

CORE_RADIUS_FRACTION = 0.5


# Core density [kg/m^3]

CORE_DENSITY = 11000.0


# Convective heat flux [W/m^2]

CONVECTIVE_HEAT_FLUX = 0.05


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

G = 6.67430e-11

MU0 = 4.0 * np.pi * 1e-7

M_PROTON = 1.67262192369e-27

K_B = 1.380649e-23


# ============================================================
# ASTRONOMICAL CONSTANTS
# ============================================================

AU = 1.495978707e11

M_SUN = 1.98847e30

R_SUN = 6.957e8

M_JUP = 1.89813e27

R_JUP = 7.1492e7

R_EARTH = 6.371e6


T_SUN = 5772.0

MU_MEAN = 0.5


# ============================================================
# BASIC NUMERIC CONVERSION
# ============================================================

def clean_numeric(value):

    """
    Convert a value to float.

    Returns NaN if conversion fails.
    """

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except (ValueError, TypeError):
        return np.nan


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    """
    Basic text normalization.

    Removes Unicode differences and normalizes whitespace.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = value.replace("−", "-")

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# PLANET NAME NORMALIZATION
# ============================================================

def normalize_name(value):

    """
    Aggressive normalization.

    Example:

        Kepler-22 b
        Kepler 22 b
        kepler22b

    all become:

        kepler22b
    """

    value = normalize_text(value)

    value = re.sub(
        r"[^a-z0-9]",
        "",
        value
    )

    return value


# ============================================================
# PLANET LETTER EXTRACTION
# ============================================================

def extract_planet_letter(name):

    """
    Extract final planetary component.

    Example:

        Kepler-22 b -> b
        TOI-700 d   -> d
    """

    if pd.isna(name):
        return ""

    name = normalize_text(name)

    match = re.search(
        r"(?:\s|-)?([bcdefghijk])$",
        name
    )

    if match:
        return match.group(1)

    return ""


# ============================================================
# NAME VARIANTS
# ============================================================

def generate_name_variants(name):

    """
    Generate several normalized representations
    of a planet name.
    """

    if pd.isna(name):
        return set()

    original = normalize_text(name)

    variants = set()

    # Original
    variants.add(original)

    # Remove spaces
    variants.add(
        original.replace(" ", "")
    )

    # Remove hyphens
    variants.add(
        original.replace("-", "")
    )

    # Remove spaces and hyphens
    variants.add(
        re.sub(
            r"[\s-]",
            "",
            original
        )
    )

    # Remove all punctuation
    variants.add(
        re.sub(
            r"[^a-z0-9]",
            "",
            original
        )
    )

    # Host + planetary letter
    match = re.match(
        r"(.+?)[\s-]+([bcdefghijk])$",
        original
    )

    if match:

        host = match.group(1)

        letter = match.group(2)

        variants.add(
            host + letter
        )

        variants.add(
            host.replace("-", "")
            + letter
        )

    return {
        normalize_name(v)
        for v in variants
        if v
    }


# ============================================================
# CSV DELIMITER DETECTION
# ============================================================

def detect_delimiter(filename):

    """
    Automatically detect CSV delimiter.
    """

    with open(
        filename,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        sample = f.read(10000)

    try:

        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=";,\t|"
        )

        return dialect.delimiter

    except csv.Error:

        # Your supplied dataset is semicolon separated.
        return ";"


# ============================================================
# LOAD INPUT FILE
# ============================================================

print()
print("=" * 70)
print("LOADING INPUT DATASET")
print("=" * 70)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Place the CSV file in the same directory "
        "as this script."
    )


delimiter = detect_delimiter(
    INPUT_FILE
)


print(
    f"\nDetected delimiter: {repr(delimiter)}"
)


input_df = pd.read_csv(
    INPUT_FILE,
    sep=delimiter,
    encoding="utf-8-sig"
)


# Clean column names

input_df.columns = (
    input_df.columns
    .astype(str)
    .str.strip()
)


print("\nInput columns:")

for column in input_df.columns:

    print(
        f"    {column}"
    )


# ============================================================
# IDENTIFY PLANET-NAME COLUMN
# ============================================================

possible_planet_columns = [

    "planet-name",
    "planet_name",
    "planet",
    "pl_name",
    "Planet",
    "Planet Name",
    "Planet_Name",
    "name",
    "Name"

]


planet_column = None


# Exact match

for candidate in possible_planet_columns:

    if candidate in input_df.columns:

        planet_column = candidate

        break


# Case-insensitive match

if planet_column is None:

    lower_columns = {

        str(column).strip().lower():
        column

        for column in input_df.columns

    }

    for candidate in possible_planet_columns:

        key = candidate.lower()

        if key in lower_columns:

            planet_column = (
                lower_columns[key]
            )

            break


if planet_column is None:

    raise ValueError(
        "\nCould not identify the planet-name column.\n\n"
        "Available columns:\n"
        +
        "\n".join(
            f"    {column}"
            for column in input_df.columns
        )
    )


print(
    f"\nUsing planet-name column: "
    f"{planet_column}"
)


# ============================================================
# EXTRACT ONLY USER'S PLANETS
# ============================================================

input_planets = (

    input_df[
        planet_column
    ]

    .dropna()

    .astype(str)

    .str.strip()

)


input_planets = [

    planet

    for planet in input_planets

    if planet != ""

]


# Preserve order and remove duplicates

input_planets = list(
    dict.fromkeys(
        input_planets
    )
)


print(
    f"\nNumber of selected planets: "
    f"{len(input_planets)}"
)


print("\nSelected planets:")

for i, planet in enumerate(
    input_planets,
    start=1
):

    print(
        f"{i:4d}. {planet}"
    )


# ============================================================
# NASA PSCompPars QUERY
# ============================================================
#
# IMPORTANT:
#
# default_flag is NOT included.
#
# NASA documentation lists default_flag in PS,
# but not PSCompPars.
#
# Therefore this query intentionally does not contain
# default_flag.
#
# ============================================================

print()
print("=" * 70)
print("NASA EXOPLANET ARCHIVE")
print("=" * 70)


print(
    "\nDownloading PSCompPars data..."
)


archive_columns = """

pl_name,
hostname,
pl_letter,
hd_name,
hip_name,
tic_id,

pl_massj,
pl_massjerr1,
pl_massjerr2,

pl_radj,
pl_radjerr1,
pl_radjerr2,

pl_orbsmax,
pl_orbsmaxerr1,
pl_orbsmaxerr2,

pl_orbper,
pl_orbpererr1,
pl_orbpererr2,

st_mass,
st_masserr1,
st_masserr2,

st_rad,
st_raderr1,
st_raderr2,

st_teff,
st_tefferr1,
st_tefferr2,

st_lum,
st_lumerr1,
st_lumerr2,

st_spectype

"""


try:

    archive_table = (
        NasaExoplanetArchive.query_criteria(

            table="pscomppars",

            select=archive_columns

        )
    )


except Exception as error:

    print()
    print(
        "NASA EXOPLANET ARCHIVE QUERY FAILED"
    )

    print()
    print(
        repr(error)
    )

    raise


archive_df = (
    archive_table.to_pandas()
)


print(
    f"\nNASA returned "
    f"{len(archive_df)} rows."
)


print(
    "\nNASA columns:"
)

for column in archive_df.columns:

    print(
        f"    {column}"
    )


# ============================================================
# IMPORTANT:
#
# PSCompPars does not contain default_flag.
#
# Therefore:
#
#     DO NOT FILTER ON default_flag.
#
# ============================================================

default_df = archive_df.copy()


# ============================================================
# BUILD LOCAL NASA LOOKUP
# ============================================================

print()
print(
    "Building NASA planet-name lookup..."
)


lookup = {}


for index, row in default_df.iterrows():

    # --------------------------------------------------------
    # Canonical NASA planet name
    # --------------------------------------------------------

    pl_name = row.get(
        "pl_name",
        ""
    )


    variants = (
        generate_name_variants(
            pl_name
        )
    )


    # --------------------------------------------------------
    # Hostname + planet letter
    # --------------------------------------------------------

    hostname = row.get(
        "hostname",
        ""
    )

    pl_letter = row.get(
        "pl_letter",
        ""
    )


    if (
        pd.notna(hostname)
        and
        pd.notna(pl_letter)
    ):

        hostname = str(
            hostname
        ).strip()

        pl_letter = str(
            pl_letter
        ).strip()


        combinations = [

            hostname + pl_letter,

            hostname + " " + pl_letter,

            hostname + "-" + pl_letter

        ]


        for combination in combinations:

            variants.update(
                generate_name_variants(
                    combination
                )
            )


    # --------------------------------------------------------
    # Save all variants
    # --------------------------------------------------------

    for variant in variants:

        if not variant:
            continue

        if variant not in lookup:

            lookup[
                variant
            ] = index


print(
    f"Lookup contains "
    f"{len(lookup)} normalized names."
)


# ============================================================
# MATCH PLANETS
# ============================================================

print()
print("=" * 70)
print("MATCHING SELECTED PLANETS TO NASA")
print("=" * 70)


matched_rows = []

unmatched_planets = []


for planet in input_planets:

    variants = (
        generate_name_variants(
            planet
        )
    )


    found_index = None


    # --------------------------------------------------------
    # METHOD 1:
    # Normalized exact match
    # --------------------------------------------------------

    for variant in variants:

        if variant in lookup:

            found_index = (
                lookup[
                    variant
                ]
            )

            break


    if found_index is not None:

        row = (
            default_df
            .loc[
                found_index
            ]
            .copy()
        )


        row[
            "input_planet"
        ] = planet


        row[
            "match_type"
        ] = "normalized_exact"


        matched_rows.append(
            row
        )


        print(
            f"[MATCH] {planet}"
            f" --> "
            f"{row['pl_name']}"
        )


    else:

        unmatched_planets.append(
            planet
        )


        print(
            f"[NOT FOUND LOCALLY] "
            f"{planet}"
        )


# ============================================================
# NASA TARGET-BY-TARGET FALLBACK
# ============================================================
#
# This is the important new part.
#
# For every planet that was not found in our local lookup,
# ask NASA directly using query_object().
#
# query_object() supports regularization / alias handling.
#
# ============================================================

if len(unmatched_planets) > 0:

    print()
    print("=" * 70)
    print("NASA DIRECT NAME LOOKUP")
    print("=" * 70)


    still_unmatched = []


    for planet in unmatched_planets:

        print()
        print(
            f"Searching NASA directly for: "
            f"{planet}"
        )


        try:

            target_table = (
                NasaExoplanetArchive.query_object(

                    planet,

                    table="pscomppars",

                    regularize=True

                )
            )


        except Exception as error:

            print(
                f"  NASA lookup error: "
                f"{error}"
            )

            still_unmatched.append(
                planet
            )

            continue


        target_df = (
            target_table.to_pandas()
        )


        if len(target_df) == 0:

            print(
                "  No NASA record found."
            )

            still_unmatched.append(
                planet
            )

            continue


        # ----------------------------------------------------
        # If several records are returned, try to select
        # the best matching canonical planet name.
        # ----------------------------------------------------

        target_index = None


        input_variants = (
            generate_name_variants(
                planet
            )
        )


        for i, target_row in (
            target_df.iterrows()
        ):

            target_name = target_row.get(
                "pl_name",
                ""
            )


            target_variants = (
                generate_name_variants(
                    target_name
                )
            )


            if (
                input_variants
                &
                target_variants
            ):

                target_index = i

                break


        # If no normalized match was found,
        # use the first returned record.

        if target_index is None:

            target_index = (
                target_df.index[0]
            )


        row = (
            target_df
            .loc[
                target_index
            ]
            .copy()
        )


        row[
            "input_planet"
        ] = planet


        row[
            "match_type"
        ] = "NASA_query_object"


        matched_rows.append(
            row
        )


        print(
            f"  [NASA MATCH] "
            f"{planet}"
            f" --> "
            f"{row.get('pl_name', '')}"
        )


    unmatched_planets = (
        still_unmatched
    )


# ============================================================
# CREATE MATCHED DATAFRAME
# ============================================================

if len(matched_rows) > 0:

    matched_df = pd.DataFrame(
        matched_rows
    )

else:

    matched_df = pd.DataFrame()


# Remove duplicate input planets

if len(matched_df) > 0:

    matched_df = (
        matched_df
        .drop_duplicates(
            subset=[
                "input_planet"
            ]
        )
    )


# ============================================================
# CREATE UNMATCHED DATAFRAME
# ============================================================

unmatched_df = pd.DataFrame({

    "input_planet":
        unmatched_planets

})


# ============================================================
# SAVE MATCH FILES
# ============================================================

matched_df.to_csv(
    MATCHED_FILE,
    index=False,
    float_format="%.15g",
    na_rep=""
)


unmatched_df.to_csv(
    UNMATCHED_FILE,
    index=False,
    na_rep=""
)


# ============================================================
# MATCHING SUMMARY
# ============================================================

print()
print("=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)


print(
    f"\nSelected planets: "
    f"{len(input_planets)}"
)


print(
    f"NASA matched: "
    f"{len(matched_df)}"
)


print(
    f"NASA unmatched: "
    f"{len(unmatched_df)}"
)


if len(input_planets) > 0:

    match_rate = (

        len(matched_df)
        /
        len(input_planets)
        *
        100.0

    )

    print(
        f"Match rate: "
        f"{match_rate:.2f}%"
    )


if len(unmatched_df) > 0:

    print(
        "\nStill unmatched:"
    )

    for planet in unmatched_planets:

        print(
            f"    {planet}"
        )


# ============================================================
# STOP IF NOTHING MATCHED
# ============================================================

if len(matched_df) == 0:

    raise RuntimeError(
        "\nNo selected planets could be matched "
        "to NASA Exoplanet Archive."
    )


# ============================================================
# CONVERT NASA NUMERIC PARAMETERS
# ============================================================

numeric_columns = [

    "pl_massj",
    "pl_massjerr1",
    "pl_massjerr2",

    "pl_radj",
    "pl_radjerr1",
    "pl_radjerr2",

    "pl_orbsmax",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",

    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",

    "st_mass",
    "st_masserr1",
    "st_masserr2",

    "st_rad",
    "st_raderr1",
    "st_raderr2",

    "st_teff",
    "st_tefferr1",
    "st_tefferr2",

    "st_lum",
    "st_lumerr1",
    "st_lumerr2"

]


for column in numeric_columns:

    if column in matched_df.columns:

        matched_df[
            column
        ] = pd.to_numeric(

            matched_df[
                column
            ],

            errors="coerce"

        )


# ============================================================
# STELLAR MASS-LOSS MODEL
# ============================================================

def stellar_mass_loss(
    M_star_solar
):

    """
    Parameterized stellar mass-loss rate:

        dotM_star =
            dotM_sun *
            (M_star / M_sun)^alpha *
            activity_factor
    """

    if not np.isfinite(
        M_star_solar
    ):

        return np.nan


    if M_star_solar <= 0:

        return np.nan


    return (

        M_SUN_LOSS

        *

        (
            M_star_solar
            ** MASS_LOSS_EXPONENT
        )

        *

        ACTIVITY_FACTOR

    )


# ============================================================
# STELLAR WIND TEMPERATURE
# ============================================================

def stellar_wind_temperature(
    T_eff
):

    """
    Adopted approximation:

        T_sw = 0.8 T_eff
    """

    if not np.isfinite(
        T_eff
    ):

        return np.nan


    if T_eff <= 0:

        return np.nan


    return 0.8 * T_eff


# ============================================================
# SOUND SPEED
# ============================================================

def sound_speed(
    T_sw
):

    if not np.isfinite(
        T_sw
    ):

        return np.nan


    if T_sw <= 0:

        return np.nan


    return np.sqrt(

        K_B
        * T_sw

        /

        (
            MU_MEAN
            * M_PROTON
        )

    )


# ============================================================
# PARKER WIND VELOCITY
# ============================================================

def parker_velocity(
    r,
    R_star,
    M_star,
    T_sw
):

    """
    Isothermal Parker wind solution.
    """

    if not all(
        np.isfinite(x)
        for x in [
            r,
            R_star,
            M_star,
            T_sw
        ]
    ):

        return np.nan


    if (
        r <= R_star
        or
        R_star <= 0
        or
        M_star <= 0
        or
        T_sw <= 0
    ):

        return np.nan


    a = sound_speed(
        T_sw
    )


    if not np.isfinite(a):

        return np.nan


    r_c = (

        G
        * M_star

        /

        (
            2.0
            * a**2
        )

    )


    if r_c <= 0:

        return np.nan


    x = r / r_c


    rhs = (

        4.0 * np.log(x)

        +

        4.0 / x

        -

        3.0

    )


    def equation(y):

        return (

            y
            -
            np.log(y)
            -
            rhs

        )


    # Subsonic branch

    if r < r_c:

        try:

            y = brentq(

                equation,

                1e-12,

                1.0 - 1e-10

            )

        except ValueError:

            return np.nan


    # Supersonic branch

    else:

        try:

            y = brentq(

                equation,

                1.0 + 1e-10,

                max(
                    2.0,
                    rhs + 10.0
                )

            )

        except ValueError:

            return np.nan


    if y <= 0:

        return np.nan


    return (
        a
        * np.sqrt(y)
    )


# ============================================================
# MINIMUM MAGNETIC DIPOLE MOMENT
# ============================================================

def minimum_magnetic_moment(
    R_s,
    Mdot,
    v_eff,
    r_orb
):

    """
    Minimum dipole moment required to sustain
    the adopted magnetopause standoff distance.
    """

    if not all(
        np.isfinite(x)
        for x in [
            R_s,
            Mdot,
            v_eff,
            r_orb
        ]
    ):

        return np.nan


    if any(
        x <= 0
        for x in [
            R_s,
            Mdot,
            v_eff,
            r_orb
        ]
    ):

        return np.nan


    return (

        R_s**3

        *

        np.sqrt(

            (

                2.0
                * np.pi
                * Mdot
                * kappa
                * v_eff

            )

            /

            (

                MU0
                * f0**2
                * r_orb**2

            )

        )

    )


# ============================================================
# DYNAMO MAGNETIC MOMENT
# ============================================================

def dynamo_magnetic_moment(
    R_p
):

    """
    Estimate planetary dynamo magnetic moment
    using the adopted scaling model.
    """

    if not np.isfinite(
        R_p
    ):

        return np.nan


    if R_p <= 0:

        return np.nan


    # Core radius

    r_c = (

        CORE_RADIUS_FRACTION
        * R_p

    )


    # CMB field scaling

    B_cmb = (

        dynamo_c

        *

        np.sqrt(
            MU0
        )

        *

        (
            CORE_DENSITY
            ** (1.0 / 6.0)
        )

        *

        (
            CONVECTIVE_HEAT_FLUX
            ** (1.0 / 3.0)
        )

    )


    # Surface dipole field

    B_p = (

        B_cmb

        *

        (
            r_c / R_p
        )**3

    )


    # Magnetic dipole moment

    M_dyn = (

        4.0
        * np.pi
        * B_p
        * R_p**3

        /

        MU0

    )


    return M_dyn


# ============================================================
# MAIN SCIENTIFIC CALCULATION
# ============================================================

print()
print("=" * 70)
print("RUNNING MAGNETIC HABITABILITY CALCULATIONS")
print("=" * 70)


results = []


for _, row in matched_df.iterrows():

    planet = row[
        "input_planet"
    ]


    print()
    print(
        f"Processing: {planet}"
    )


    # ========================================================
    # PLANET PARAMETERS
    # ========================================================

    M_p_jup = clean_numeric(
        row.get(
            "pl_massj",
            np.nan
        )
    )


    R_p_jup = clean_numeric(
        row.get(
            "pl_radj",
            np.nan
        )
    )


    a_AU = clean_numeric(
        row.get(
            "pl_orbsmax",
            np.nan
        )
    )


    P_days = clean_numeric(
        row.get(
            "pl_orbper",
            np.nan
        )
    )


    # ========================================================
    # STELLAR PARAMETERS
    # ========================================================

    M_star_solar = clean_numeric(
        row.get(
            "st_mass",
            np.nan
        )
    )


    R_star_solar = clean_numeric(
        row.get(
            "st_rad",
            np.nan
        )
    )


    T_eff = clean_numeric(
        row.get(
            "st_teff",
            np.nan
        )
    )




    spectral_type = row.get(
        "st_spectype",
        ""
    )


    # ========================================================
    # UNIT CONVERSIONS
    # ========================================================

    if np.isfinite(
        M_p_jup
    ):

        M_p = (
            M_p_jup
            * M_JUP
        )

    else:

        M_p = np.nan


    if np.isfinite(
        R_p_jup
    ):

        R_p = (
            R_p_jup
            * R_JUP
        )

    else:

        R_p = np.nan


    if np.isfinite(
        a_AU
    ):

        a = (
            a_AU
            * AU
        )

    else:

        a = np.nan


    if np.isfinite(
        M_star_solar
    ):

        M_star = (
            M_star_solar
            * M_SUN
        )

    else:

        M_star = np.nan


    if np.isfinite(
        R_star_solar
    ):

        R_star = (
            R_star_solar
            * R_SUN
        )

    else:

        R_star = np.nan


    # ========================================================
    # HABITABLE ZONE
    # ========================================================
    #
    # IMPORTANT:
    # HZ values are NOT read from the input CSV.
    # All observational stellar quantities come from NASA PSCompPars.
    #
    # st_lum is log10(L_star/L_sun), so convert it first:
    #
    #     L_star/L_sun = 10**st_lum
    #
    # The HZ is then calculated using the Kopparapu et al. polynomial
    # prescription from the NASA stellar effective temperature.
    # ========================================================

    # NASA archive luminosity is log10(L/L_sun).
    log_L_star = clean_numeric(
        row.get(
            "st_lum",
            np.nan
        )
    )

    if np.isfinite(log_L_star):
        L_star = 10.0 ** log_L_star
    else:
        L_star = np.nan

    def kopparapu_hz(T_eff_K):
        """
        Conservative habitable-zone limits following
        Kopparapu et al. polynomial coefficients.

        Returns:
            (inner_AU, outer_AU)
        """
        if not np.isfinite(T_eff_K) or T_eff_K <= 0:
            return np.nan, np.nan

        # Effective stellar temperature relative to 5780 K.
        T_prime = T_eff_K - 5780.0

        # Solar effective flux coefficients for:
        # runaway greenhouse (inner) and maximum greenhouse (outer).
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
            return np.nan, np.nan

        # For L/L_sun and S_eff/S_sun:
        # d(AU) = sqrt(L/S_eff)
        inner = np.sqrt(L_star / S_inner) if np.isfinite(L_star) and L_star > 0 else np.nan
        outer = np.sqrt(L_star / S_outer) if np.isfinite(L_star) and L_star > 0 else np.nan

        return inner, outer

    hz_inner, hz_outer = kopparapu_hz(T_eff)

    in_hz = False

    if all(
        np.isfinite(x)
        for x in [
            a_AU,
            hz_inner,
            hz_outer
        ]
    ):

        in_hz = (
            hz_inner
            <=
            a_AU
            <=
            hz_outer
        )

    # ========================================================
    # STELLAR WIND
    # ========================================================

    T_sw = stellar_wind_temperature(
        T_eff
    )


    Mdot = stellar_mass_loss(
        M_star_solar
    )


    v_eff = parker_velocity(

        a,

        R_star,

        M_star,

        T_sw

    )


    # ========================================================
    # STELLAR WIND NUMBER DENSITY
    # ========================================================

    if (

        np.isfinite(Mdot)

        and
        np.isfinite(a)

        and
        np.isfinite(v_eff)

        and
        v_eff > 0

    ):

        eta = (

            Mdot
            * kappa

            /

            (
                4.0
                * np.pi
                * a**2
                * M_PROTON
                * v_eff
            )

        )

    else:

        eta = np.nan


    # ========================================================
    # DYNAMIC PRESSURE
    # ========================================================

    if (

        np.isfinite(eta)

        and
        np.isfinite(v_eff)

    ):

        P_dynamic = (

            eta
            * M_PROTON
            * v_eff**2

        )

    else:

        P_dynamic = np.nan


    # ========================================================
    # THERMAL PRESSURE
    # ========================================================

    if (

        np.isfinite(eta)

        and
        np.isfinite(T_sw)

    ):

        P_thermal = (

            eta
            * K_B
            * T_sw

        )

    else:

        P_thermal = np.nan


    # ========================================================
    # DYNAMO MAGNETIC MOMENT
    # ========================================================

    M_dyn = dynamo_magnetic_moment(
        R_p
    )


    # ========================================================
    # STANDOFF SCENARIOS
    # ========================================================

    for standoff_factor in STANDOFF_FACTORS:

        # ----------------------------------------------------
        # Magnetopause radius
        # ----------------------------------------------------

        if np.isfinite(
            R_p
        ):

            R_s = (

                standoff_factor
                * R_p

            )

        else:

            R_s = np.nan


        # ----------------------------------------------------
        # Required magnetic moment
        # ----------------------------------------------------

        M_min = (
            minimum_magnetic_moment(

                R_s,

                Mdot,

                v_eff,

                a

            )
        )


        # ----------------------------------------------------
        # Magnetic habitability coefficient
        # ----------------------------------------------------

        if (

            np.isfinite(M_dyn)

            and
            np.isfinite(M_min)

            and
            M_min > 0

        ):

            M_hab = (

                M_dyn
                /
                M_min

            )

        else:

            M_hab = np.nan


        # ----------------------------------------------------
        # Magnetic sufficiency
        # ----------------------------------------------------

        if np.isfinite(
            M_hab
        ):

            magnetically_sufficient = (
                M_hab >= 1.0
            )

        else:

            magnetically_sufficient = False


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            # ================================================
            # IDENTIFICATION
            # ================================================

            "planet_name":
                planet,

            "nasa_pl_name":
                row.get(
                    "pl_name",
                    ""
                ),

            "hostname":
                row.get(
                    "hostname",
                    ""
                ),

            "spectral_type":
                spectral_type,

            "match_type":
                row.get(
                    "match_type",
                    ""
                ),


            # ================================================
            # PLANETARY PARAMETERS
            # ================================================

            "planet_mass_Mjup":
                M_p_jup,

            "planet_radius_Rjup":
                R_p_jup,

            "semi_major_axis_AU":
                a_AU,

            "orbital_period_days":
                P_days,


            # ================================================
            # STELLAR PARAMETERS
            # ================================================

            "stellar_mass_Msun":
                M_star_solar,

            "stellar_radius_Rsun":
                R_star_solar,

            "stellar_teff_K":
                T_eff,

            "stellar_luminosity_log10_Lsun":
                log_L_star,

            "stellar_luminosity_Lsun":
                L_star,


            # ================================================
            # HABITABLE ZONE
            # ================================================

            "hz_inner_AU":
                hz_inner,

            "hz_outer_AU":
                hz_outer,

            "in_habitable_zone":
                in_hz,


            # ================================================
            # STELLAR WIND
            # ================================================

            "stellar_wind_temperature_K":
                T_sw,

            "stellar_mass_loss_kg_s":
                Mdot,

            "stellar_wind_velocity_m_s":
                v_eff,

            "stellar_wind_density_m-3":
                eta,

            "stellar_wind_dynamic_pressure_Pa":
                P_dynamic,

            "stellar_wind_thermal_pressure_Pa":
                P_thermal,


            # ================================================
            # MAGNETOSPHERE
            # ================================================

            "standoff_factor_Rs_Rp":
                standoff_factor,

            "standoff_distance_m":
                R_s,

            "minimum_magnetic_moment_A_m2":
                M_min,


            # ================================================
            # DYNAMO
            # ================================================

            "dynamo_magnetic_moment_A_m2":
                M_dyn,


            # ================================================
            # FINAL METRIC
            # ================================================

            "magnetic_habitability_coefficient":
                M_hab,

            "magnetically_sufficient":
                magnetically_sufficient

        })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False,
    float_format="%.15g",
    na_rep=""
)

# Re-read the file as text to ensure decimal values were written
# literally and are not being altered by a dataframe/display layer.
print()
print("CSV numeric export check:")
for col in [
    "planet_mass_Mjup",
    "planet_radius_Rjup",
    "semi_major_axis_AU",
    "stellar_mass_Msun",
    "stellar_radius_Rsun",
    "stellar_teff_K",
    "stellar_luminosity_log10_Lsun",
    "stellar_luminosity_Lsun",
    "hz_inner_AU",
    "hz_outer_AU"
]:
    if col in results_df.columns:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            print(f"    {col}: min={vals.min():.15g}, max={vals.max():.15g}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CALCULATION COMPLETE")
print("=" * 70)


print(
    f"\nSelected planets: "
    f"{len(input_planets)}"
)


print(
    f"NASA matched planets: "
    f"{len(matched_df)}"
)


print(
    f"NASA unmatched planets: "
    f"{len(unmatched_df)}"
)


print(
    f"Calculation rows: "
    f"{len(results_df)}"
)


print(
    f"Standoff scenarios per planet: "
    f"{len(STANDOFF_FACTORS)}"
)


# ============================================================
# VALID CALCULATIONS
# ============================================================

valid_results = results_df[
    results_df[
        "magnetic_habitability_coefficient"
    ].notna()
].copy()


print(
    f"\nValid magnetic calculations: "
    f"{len(valid_results)}"
)


# ============================================================
# MAGNETICALLY SUFFICIENT
# ============================================================

if len(valid_results) > 0:

    sufficient = valid_results[
        valid_results[
            "magnetically_sufficient"
        ]
    ]


    print(
        "\nMagnetically sufficient cases:"
    )


    print(
        f"    {len(sufficient)} / "
        f"{len(valid_results)}"
    )


    # --------------------------------------------------------
    # Maximum M_hab for each planet
    # --------------------------------------------------------

    planet_summary = (

        valid_results

        .groupby(
            "planet_name"
        )

        [
            "magnetic_habitability_coefficient"
        ]

        .max()

        .sort_values(
            ascending=False
        )

    )


    print()
    print(
        "Top planets by maximum M_hab:"
    )


    print(
        planet_summary.head(20)
    )


# ============================================================
# HABITABLE-ZONE SUMMARY
# ============================================================

if len(valid_results) > 0:

    hz_results = valid_results[
        valid_results[
            "in_habitable_zone"
        ] == True
    ]


    print()
    print(
        "Planets inside supplied HZ:"
    )


    print(
        f"    {hz_results['planet_name'].nunique()}"
    )


# ============================================================
# SPECTRAL TYPE SUMMARY
# ============================================================

if (

    len(valid_results) > 0

    and

    "spectral_type"
    in valid_results.columns

):

    valid_results[
        "spectral_group"
    ] = (

        valid_results[
            "spectral_type"
        ]

        .fillna("")

        .astype(str)

        .str.upper()

        .str[0]

    )


    spectral_summary = (

        valid_results

        .groupby(
            "spectral_group"
        )

        .agg(

            planets=(
                "planet_name",
                "nunique"
            ),

            median_Mhab=(
                "magnetic_habitability_coefficient",
                "median"
            ),

            median_wind_pressure=(
                "stellar_wind_dynamic_pressure_Pa",
                "median"
            ),

            magnetically_sufficient=(
                "magnetically_sufficient",
                "sum"
            )

        )

    )


    print()
    print("=" * 70)
    print("SPECTRAL TYPE SUMMARY")
    print("=" * 70)


    print(
        spectral_summary
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)


print(
    f"\n1. {MATCHED_FILE}"
)

print(
    f"2. {UNMATCHED_FILE}"
)

print(
    f"3. {RESULT_FILE}"
)


print()
print(
    "Finished successfully."
)