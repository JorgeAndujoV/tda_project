#!/usr/bin/env python3
"""
verify_integrity.py
───────────────────
Re-derives dataset2_limpio.csv and dataset2_limpio_ohe.csv from the raw
source (BASE FINAL-Tabla 1.csv) using the same transformations as
exploracion.ipynb, then compares the results column-by-column.

Usage
-----
    python verify_integrity.py          # uses default paths in ./data/
    python verify_integrity.py --data-dir /path/to/data

Exit code 0 → all checks passed.
Exit code 1 → at least one check failed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── colour helpers ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


# ── column dictionary from the notebook (needed to classify columns) ───────
DESCRIPCION_COLUMNAS = {
    "n": "Identificador encuestado (numerico)",
    "Fecha Encuesta": "Fecha en que se realizó la encuesta (fecha)",
    "Edad madre": "Edad de la madre al momento de la encuesta (numerico)",
    "Nacionalidad": "Nacionalidad de la madre (nominal 1-7)",
    "Agrupación nacionalidad": "nominal (1-2)",
    "Región": "Región de residencia (nominal 1-16)",
    "Educación": "Nivel educacional de la madre (nominal 1-7)",
    "FN hijo": "Fecha de nacimiento de ultimo hijo (fecha)",
    "Período": "no se sabe nominal (1-2)",
    "N° embarazo": "numero del ultimo embarazo (numerico)",
    "Sexo hijo": "nominal (1-3)",
    "PN hijo (g)": "peso de hijo al nacer en gramos (numerico)",
    "EG hijo (sem)": "edad gestacional del hijo en semanas (nuemrico)",
    "KG inicio mamá": "peso en kilogramos de la madre al iniciar el embarazo (numerico)",
    "KG fin mamá": "peso en kilogramos de la madre al terminar su embarazo (numerico)",
    "Dif peso mamá": "Diferencia de peso de madre en kilogramos (numerico)",
    "Estatura mamá": "estatura mama en centimetros (numerico)",
    "IMC antes": "Indice de masa corporal de la madre antes del embarazo (numerico)",
    "IMC después": "Indice de masa corporal de la madre despues del embarazo (numerico)",
    "Dif IMC": "diferencia en IMC madre (numerico)",
    "Condición mamá": "condiciones a las que pudo estar expuesta la madre antes o durante el embarazo (nominal 1-3)",
    "¿Hijo nace c/problema de salud?": "responde si el hijo nacio con algun problema de salud (binario)",
    "Patología RN": "si la respuesta anterior fue si especifica la condición (nominal 0-10 con 0=no)",
    "¿Consume suplementos y/o multivitamínico de AF en embarazo?": "responde si se consumio algun tipo de suplemento de acido folico o multivitaminico durante el embarazo (binario)",
    "¿Consume SAF?": "responde si se consumen suplementos de acido folico (binario)",
    "¿Consume SAF+OSAF?": "responde si se consumen sumplementos de acido folico y otros suplementos de acido folico (binario)",
    "Código SAF": "especifica que suplemento se consumio (nominal 0-42 con 0 = no consumio)",
    "mgAF/cáp SAF+OSAF": "miligramos de acido folico por capsula de suplementos (numerico)",
    "¿Cuántos días de SAF?": "responde cuantos dias a la semana se consumio suplemento de acido folico (numerico 1-7)",
    "¿Cuántas cáp SAF?": "cantidad de capsulas de suplemento de acido folico consumido al dia (nominal 1-3)",
    "mgAF/día SAF": "miligramos de acido folico al dia en suplementos de acido folico (numerico)",
    "¿Consumió MAF?": "responde si consumio Multivitamínicos con AF (binario)",
    "¿Consumió MAF completo?": "responde si consumio Multivitamínicos con AF  completo (binario)",
    "Código MAF": "especifica multivitaminico con acido folico (nominal 0-42 con 0 = no consumio)",
    "mgAF/cáp MAF": "miligramos de acido folico por capsula de multivtamincio (numerico)",
    "¿Cuántos días de MAF?": "cuantos dias a la semana se consumieron multivitaminicos con acido folico (numerico 1-7)",
    "¿Cuántas cáp MAF?": "capsulas de multivitaminicos con AF consumidas al dia (nominal 1-3)",
    "mgAF/día MAF 1°T": "miligramos de acido folico al dia del multivitaminico en primer trimestre (numerico)",
    "Consumo OSAF 1° toma": "consumo de otros suplementos de AF primera toma (binario)",
    "Consumo OSAF 2° toma": "consumo de otros suplementos de AF segunda toma (binario)",
    "Consumo OSAF completo": "consumio otros suplementos de AF completo (binario)",
    "Código OSAF": "especifica el otro suplemento de acido folico (nominal 0-42 con 0 = no consumio)",
    "mgAF/cáp OSAF": "miligramos de acido folico por capsula de otros suplementos de AF (numerico)",
    "¿Cuántos días de OSAF?": "dias a la semana de consumo de otros suplementos de AF (numerico 1-7)",
    "¿Cuántas cáp OSAF?": "capsulas de otros suplementos de AF al dia (nominal 1-3)",
    "mgAF/día OSAF 1°T": "mg de AF por dia de OSAF en primer trimestre (numerico)",
    "mgAF/día SAF+OSAF 2°T": "mg de AF totales al dia en segundo trimestre (numerico)",
    "mgAF/día Suplementos y multivitamínico 1°T": "mg de AF totales al dia en primer trimestre (numerico)",
    "mg/día DFE suplementos y multivitamínico 1°T": "DFE al dia en primer trimestre (numerico)",
    "Período consumo SAF+OSAF": "periodo en el que se consumieron SAF y OSAF (nominal 1-4)",
    "Días consumo MAF": "días de consumo de MAF (numerico)",
    "mgAF/día período MAF": "mg de AF por dia en periodo MAF (numerico)",
    "Días consumo suplementosAF": "dias de consumo de suplementos de AF (numerico)",
    "mgAF/día total período SAF+OSAF": "mg AF totales al dia en periodo SAF+OSAF (numerico)",
    "TOTAL AF mg suplementos y multivitamínicos período ": "total de mg de AF en suplementos y multivitaminicos en periodo (numerico)",
    "Código consumo SIN": "sin info (nominal 0-42)",
    "Código alimentos": "codigo alimentos (nominal 1-4)",
    "¿Consumió pan?": "consumio pan (binario)",
    "Código pan": "tipo de pan consumido (nominal 0-15)",
    "mgAF/unidad pan": "mg de AF por unidad de pan (numerico)",
    "Código pan/día consumido": "codigo pan por dia consumido (nominal 0-4)",
    "mg/d AF total pan": "mg de AF total por pan al dia (numerico)",
    "mg/d DFE total pan": "DFE total de pan al dia (numerico)",
    "Total mg/d AF  suple y pan": "total mg AF al dia por suplementos y pan (numerico)",
    "Total mg/d DFE suple y pan": "total DFE al dia por suplementos y pan (numerico)",
    "Consumo suple y pan y pastas (si/no)": "consumo de suplementos pan y pastas (binario)",
}


# ── replicate the notebook's column classification logic ───────────────────
def classify_columns(df: pd.DataFrame) -> tuple[list, list, list, list]:
    """Return (nominales, binarias, numericas, fechas) exactly as the notebook does."""
    nominales, binarias, numericas, fechas = [], [], [], []
    for col, desc in DESCRIPCION_COLUMNAS.items():
        if col not in df.columns:
            continue
        d = desc.lower()
        if "nominal" in d:
            nominales.append(col)
        elif "binari" in d:
            binarias.append(col)
        elif "fecha" in d:
            fechas.append(col)
        elif "numerico" in d or "numérico" in d:
            numericas.append(col)
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                numericas.append(col)
    return nominales, binarias, numericas, fechas


# ── main checks ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify data-cleaning integrity")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing the three CSV files",
    )
    args = parser.parse_args()

    raw_path = args.data_dir / "BASE FINAL-Tabla 1.csv"
    clean_path = args.data_dir / "dataset2_limpio.csv"
    ohe_path = args.data_dir / "dataset2_limpio_ohe.csv"

    for p in (raw_path, clean_path, ohe_path):
        if not p.exists():
            fail(f"File not found: {p}")
            return 1

    failures = 0

    # ── Load all three files ────────────────────────────────────────────────
    raw = pd.read_csv(raw_path)
    saved_clean = pd.read_csv(clean_path, index_col=0)
    saved_ohe = pd.read_csv(ohe_path, index_col=0)

    # ── Re-derive the cleaned dataset (same steps as the notebook) ──────────
    derived_clean = raw.fillna(0)
    derived_clean = derived_clean.replace({",": ""}, regex=True)

    # Classify columns and coerce numerics (notebook cell 12)
    nominales, binarias, numericas, fechas = classify_columns(derived_clean)
    for col in numericas:
        derived_clean[col] = pd.to_numeric(derived_clean[col], errors="coerce")

    # Re-derive OHE
    for col in nominales:
        derived_clean[col] = derived_clean[col].astype(str)
    derived_ohe = pd.get_dummies(derived_clean, columns=nominales, dtype=int)

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{BOLD}{CYAN}═══  Data Integrity Verification  ═══{RESET}\n")

    # ── 1. Row-count preservation ───────────────────────────────────────────
    print(f"{BOLD}1. Row-count preservation{RESET}")
    if len(raw) == len(saved_clean) == len(saved_ohe):
        ok(f"All files have {len(raw)} rows")
    else:
        fail(
            f"Row mismatch — raw={len(raw)}, clean={len(saved_clean)}, ohe={len(saved_ohe)}"
        )
        failures += 1

    # ── 2. Column preservation (raw → clean) ───────────────────────────────
    print(f"\n{BOLD}2. Column preservation (raw → clean){RESET}")
    missing_in_clean = set(raw.columns) - set(saved_clean.columns)
    extra_in_clean = set(saved_clean.columns) - set(raw.columns)
    if not missing_in_clean:
        ok("All raw columns present in cleaned file")
    else:
        fail(f"Columns lost: {missing_in_clean}")
        failures += 1
    if extra_in_clean:
        warn(f"Extra columns in cleaned file: {extra_in_clean}")

    # ── 3. No NaN/null values in cleaned file ──────────────────────────────
    print(f"\n{BOLD}3. Null handling{RESET}")
    null_counts_clean = saved_clean.isnull().sum()
    cols_with_nulls = null_counts_clean[null_counts_clean > 0]
    if cols_with_nulls.empty:
        ok("No NaN/null values in dataset2_limpio.csv")
    else:
        # The notebook fills with 0 then removes commas; pd.to_numeric(errors='coerce')
        # can re-introduce NaN — check if that's expected.
        warn(
            f"{len(cols_with_nulls)} column(s) still have NaN after cleaning "
            f"(may be from to_numeric coercion):"
        )
        for c, n in cols_with_nulls.items():
            print(f"        {c}: {n} NaN")

    null_counts_ohe = saved_ohe.isnull().sum()
    cols_with_nulls_ohe = null_counts_ohe[null_counts_ohe > 0]
    if cols_with_nulls_ohe.empty:
        ok("No NaN/null values in dataset2_limpio_ohe.csv")
    else:
        warn(
            f"{len(cols_with_nulls_ohe)} column(s) still have NaN in OHE file:"
        )
        for c, n in cols_with_nulls_ohe.items():
            print(f"        {c}: {n} NaN")

    # ── 4. Value fidelity: compare derived vs saved cleaned dataset ────────
    print(f"\n{BOLD}4. Value fidelity — derived vs saved (cleaned){RESET}")
    common_cols = sorted(set(derived_clean.columns) & set(saved_clean.columns))
    mismatched_cols_clean = []
    for col in common_cols:
        s1 = derived_clean[col].reset_index(drop=True)
        s2 = saved_clean[col].reset_index(drop=True)
        # Align dtypes before comparing
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            if not np.allclose(
                s1.fillna(-999).values,
                s2.fillna(-999).values,
                atol=1e-6,
                equal_nan=True,
            ):
                mismatched_cols_clean.append(col)
        else:
            if not s1.astype(str).equals(s2.astype(str)):
                mismatched_cols_clean.append(col)

    if not mismatched_cols_clean:
        ok(f"All {len(common_cols)} common columns match exactly")
    else:
        fail(f"{len(mismatched_cols_clean)} column(s) differ between derived and saved:")
        for c in mismatched_cols_clean:
            s1 = derived_clean[c].reset_index(drop=True)
            s2 = saved_clean[c].reset_index(drop=True)
            diff_mask = s1.astype(str) != s2.astype(str)
            n_diff = diff_mask.sum()
            print(f"        {c}: {n_diff} differing values")
            # Show first 3 examples
            idxs = diff_mask[diff_mask].index[:3]
            for idx in idxs:
                print(f"            row {idx}: derived={s1.iloc[idx]!r}  saved={s2.iloc[idx]!r}")
        failures += 1

    # ── 5. Value fidelity: compare derived vs saved OHE dataset ────────────
    print(f"\n{BOLD}5. Value fidelity — derived vs saved (OHE){RESET}")
    common_cols_ohe = sorted(set(derived_ohe.columns) & set(saved_ohe.columns))
    missing_ohe_cols = set(saved_ohe.columns) - set(derived_ohe.columns)
    extra_ohe_cols = set(derived_ohe.columns) - set(saved_ohe.columns)

    if missing_ohe_cols:
        warn(f"Columns in saved OHE but not derived: {missing_ohe_cols}")
    if extra_ohe_cols:
        warn(f"Columns derived but not in saved OHE: {extra_ohe_cols}")

    mismatched_cols_ohe = []
    for col in common_cols_ohe:
        s1 = derived_ohe[col].reset_index(drop=True)
        s2 = saved_ohe[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            if not np.allclose(
                s1.fillna(-999).values,
                s2.fillna(-999).values,
                atol=1e-6,
                equal_nan=True,
            ):
                mismatched_cols_ohe.append(col)
        else:
            if not s1.astype(str).equals(s2.astype(str)):
                mismatched_cols_ohe.append(col)

    if not mismatched_cols_ohe:
        ok(f"All {len(common_cols_ohe)} common OHE columns match exactly")
    else:
        fail(f"{len(mismatched_cols_ohe)} OHE column(s) differ:")
        for c in mismatched_cols_ohe[:10]:
            print(f"        {c}")
        failures += 1

    # ── 6. OHE consistency: each row sums to 1 per original nominal ────────
    print(f"\n{BOLD}6. One-hot encoding consistency{RESET}")
    ohe_issues = 0
    for nom in nominales:
        ohe_cols = [c for c in saved_ohe.columns if c.startswith(f"{nom}_")]
        if not ohe_cols:
            warn(f"No OHE columns found for '{nom}'")
            ohe_issues += 1
            continue
        row_sums = saved_ohe[ohe_cols].sum(axis=1)
        bad_rows = (row_sums != 1).sum()
        if bad_rows == 0:
            ok(f"'{nom}': all {len(ohe_cols)} dummies sum to 1 per row")
        else:
            fail(f"'{nom}': {bad_rows} rows do NOT sum to 1 (found sums: {row_sums.unique()[:5]})")
            ohe_issues += 1
    if ohe_issues:
        failures += 1

    # ── 7. Numeric-range sanity (cleaned vs raw) ───────────────────────────
    print(f"\n{BOLD}7. Numeric-range sanity (cleaned vs raw){RESET}")
    range_issues = 0
    for col in numericas:
        if col not in raw.columns or col not in saved_clean.columns:
            continue
        raw_series = pd.to_numeric(
            raw[col].astype(str).str.replace(",", ""), errors="coerce"
        )
        clean_series = pd.to_numeric(saved_clean[col], errors="coerce")
        # Compare min/max (allowing for NaN fill to 0 shifting the min)
        r_min, r_max = raw_series.min(), raw_series.max()
        c_min, c_max = clean_series.min(), clean_series.max()
        # The cleaned min can be 0 if NaN was filled; otherwise should match
        if pd.notna(r_max) and pd.notna(c_max) and not np.isclose(r_max, c_max, atol=1e-4):
            fail(f"'{col}': max changed {r_max} → {c_max}")
            range_issues += 1
        # Min may legitimately differ because of fillna(0)
        if pd.notna(r_min) and pd.notna(c_min):
            if c_min < min(r_min, 0) - 1e-4:
                fail(f"'{col}': cleaned min ({c_min}) below raw min ({r_min}) and 0")
                range_issues += 1
    if range_issues == 0:
        ok("All numeric ranges are consistent between raw and cleaned")
    else:
        failures += 1

    # ── 8. Comma removal verification ──────────────────────────────────────
    print(f"\n{BOLD}8. Comma removal verification{RESET}")
    comma_cols = []
    for col in saved_clean.columns:
        if saved_clean[col].astype(str).str.contains(",", na=False).any():
            comma_cols.append(col)
    if not comma_cols:
        ok("No commas found in any column of dataset2_limpio.csv")
    else:
        fail(f"Commas still present in: {comma_cols}")
        failures += 1

    # ── 9. Unique-ID preservation ──────────────────────────────────────────
    print(f"\n{BOLD}9. Unique ID ('n') preservation{RESET}")
    if "n" in raw.columns and "n" in saved_clean.columns:
        raw_ids = set(raw["n"].dropna().astype(int))
        clean_ids = set(saved_clean["n"].dropna().astype(int))
        lost = raw_ids - clean_ids
        added = clean_ids - raw_ids
        if not lost and not added:
            ok(f"All {len(raw_ids)} unique IDs preserved")
        else:
            if lost:
                fail(f"{len(lost)} IDs lost: {sorted(lost)[:10]}…")
                failures += 1
            if added:
                fail(f"{len(added)} spurious IDs added: {sorted(added)[:10]}…")
                failures += 1
    else:
        warn("Column 'n' not found — skipping ID check")

    # ── 10. Distribution comparison (key numeric columns) ─────────────────
    print(f"\n{BOLD}10. Distribution preservation (mean/std of key numerics){RESET}")
    key_numerics = [
        "Edad madre", "PN hijo (g)", "EG hijo (sem)",
        "IMC antes", "IMC después", "Dif IMC",
        "KG inicio mamá", "KG fin mamá", "Dif peso mamá",
        "Estatura mamá",
    ]
    dist_issues = 0
    for col in key_numerics:
        if col not in raw.columns or col not in saved_clean.columns:
            continue
        raw_s = pd.to_numeric(
            raw[col].astype(str).str.replace(",", ""), errors="coerce"
        ).dropna()
        clean_s = pd.to_numeric(saved_clean[col], errors="coerce").dropna()

        if len(raw_s) == 0 or len(clean_s) == 0:
            warn(f"'{col}': skipped (empty after coercion)")
            continue

        mean_diff = abs(raw_s.mean() - clean_s.mean())
        std_diff = abs(raw_s.std() - clean_s.std())

        # Allow a small tolerance for floating point + the effect of fillna(0)
        # on previously-NaN rows (which adds 0 values to the cleaned set)
        raw_nan_count = raw[col].isnull().sum() + (raw[col].astype(str) == "").sum()
        if raw_nan_count > 0:
            # Recalculate raw mean/std including the 0-fills
            raw_with_fill = pd.to_numeric(
                raw[col].fillna(0).astype(str).str.replace(",", ""), errors="coerce"
            )
            mean_diff_adj = abs(raw_with_fill.mean() - clean_s.mean())
            std_diff_adj = abs(raw_with_fill.std() - clean_s.std())
        else:
            mean_diff_adj = mean_diff
            std_diff_adj = std_diff

        threshold_mean = max(0.01, 0.001 * abs(raw_s.mean())) if raw_s.mean() != 0 else 0.01
        threshold_std = max(0.01, 0.001 * abs(raw_s.std())) if raw_s.std() != 0 else 0.01

        if mean_diff_adj > threshold_mean or std_diff_adj > threshold_std:
            fail(
                f"'{col}': mean diff={mean_diff_adj:.4f} (thr={threshold_mean:.4f}), "
                f"std diff={std_diff_adj:.4f} (thr={threshold_std:.4f})"
            )
            dist_issues += 1
        else:
            ok(f"'{col}': mean/std preserved (Δmean={mean_diff_adj:.6f}, Δstd={std_diff_adj:.6f})")
    if dist_issues:
        failures += 1

    # ════════════════════════════════════════════════════════════════════════
    print()
    if failures == 0:
        print(f"{BOLD}{GREEN}All checks passed — data integrity verified ✓{RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}{failures} check(s) FAILED — review issues above ✗{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
