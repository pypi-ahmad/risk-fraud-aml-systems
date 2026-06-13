"""Download and load the real OFAC Specially Designated Nationals (SDN) list.

Source: https://www.treasury.gov/ofac/downloads/sdn.csv (keyless, public).
The file is cached at ``data/raw/sdn.csv`` so notebooks re-run offline.
"""

import httpx
import pandas as pd
from loguru import logger

from aml_investigator.settings import settings

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"

# The SDN csv ships without a header row; first four of twelve columns matter here.
SDN_COLUMNS = [
    "ent_num", "sdn_name", "sdn_type", "program", "title", "call_sign",
    "vess_type", "tonnage", "grt", "vess_flag", "vess_owner", "remarks",
]


def download_sdn(force: bool = False) -> None:
    """Fetch the SDN list unless a cached copy exists."""
    path = settings.sdn_csv_path
    if path.exists() and not force:
        logger.info(f"SDN list already cached at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SDN list from {SDN_URL}")
    resp = httpx.get(SDN_URL, follow_redirects=True, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    logger.info(f"Saved {len(resp.content):,} bytes to {path}")


def load_sdn_frame() -> pd.DataFrame:
    """Parse the cached SDN csv into (ent_num, sdn_name, sdn_type, program)."""
    df = pd.read_csv(
        settings.sdn_csv_path,
        header=None,
        names=SDN_COLUMNS,
        usecols=["ent_num", "sdn_name", "sdn_type", "program"],
        na_values=["-0- ", "-0-"],
        encoding="latin-1",
    )
    df = df.dropna(subset=["sdn_name"]).reset_index(drop=True)
    df["sdn_type"] = df["sdn_type"].fillna("entity")
    return df


def load_sdn_into(con) -> int:
    """Load the SDN frame into the warehouse ``sdn`` table. Returns row count."""
    df = load_sdn_frame()
    con.execute("DELETE FROM sdn")
    con.register("sdn_df", df)
    con.execute("INSERT INTO sdn SELECT ent_num, sdn_name, sdn_type, program FROM sdn_df")
    con.unregister("sdn_df")
    return len(df)
