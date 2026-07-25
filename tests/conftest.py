import csv
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "r_reference" / "data"
GENERATOR = Path(__file__).parent / "r_reference" / "generate_reference.R"

SETUP_HINT = (
    "R-equivalence ground truth not found. These tests are optional and need the "
    "upstream R sources, which are not distributed with this package. See "
    "'Validating against the R implementation' in README.md, then run:\n"
    "  Rscript tests/r_reference/generate_reference.R"
)


def pytest_addoption(parser):
    group = parser.getgroup("rrho2")
    group.addoption(
        "--run-r-comparison",
        action="store_true",
        default=False,
        help=(
            "Require the R-equivalence tests to run, and fail instead of skip if "
            "the ground truth is missing. Without this flag they run when the "
            "ground truth is present and skip when it is not."
        ),
    )
    group.addoption(
        "--no-r-comparison",
        action="store_true",
        default=False,
        help="Skip the R-equivalence tests even if the ground truth is present.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_r_reference: needs ground truth generated from the upstream R "
        "package (see README.md)",
    )


def reference_data_available() -> bool:
    return DATA_DIR.is_dir() and any(DATA_DIR.glob("*_hypermat.tsv"))


def pytest_collection_modifyitems(config, items):
    """Gate the R-comparison tests on the ground truth being present."""
    if config.getoption("--no-r-comparison"):
        skip = pytest.mark.skip(reason="disabled by --no-r-comparison")
        for item in items:
            if "requires_r_reference" in item.keywords:
                item.add_marker(skip)
        return

    if reference_data_available():
        return

    if config.getoption("--run-r-comparison"):
        # Hard error, not a skip: the caller explicitly asked for these tests, so
        # a green run would be a false pass. Exits with pytest's USAGE_ERROR (4).
        config._rrho2_required_but_missing = True
        raise pytest.UsageError(f"--run-r-comparison was given but the {SETUP_HINT}")

    skip = pytest.mark.skip(reason=SETUP_HINT)
    for item in items:
        if "requires_r_reference" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Make a silent skip impossible to miss.

    R/ is gitignored, so the default state of a fresh clone is "no ground truth".
    Say so loudly rather than reporting a green run that skipped a third of the
    suite.
    """
    if config.getoption("--no-r-comparison") or reference_data_available():
        return
    # --run-r-comparison already raised a UsageError explaining the same thing.
    if getattr(config, "_rrho2_required_but_missing", False):
        return
    terminalreporter.write_sep("=", "R-equivalence tests were SKIPPED", yellow=True)
    terminalreporter.write_line(SETUP_HINT)


def read_gene_list(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype="<U1")
    text = path.read_text().splitlines()
    return np.array([line for line in text if line != ""], dtype=object)


def read_matrix(path: Path) -> np.ndarray:
    """Read a whitespace-separated matrix written by R, mapping NA to nan."""
    rows = []
    for line in path.read_text().splitlines():
        rows.append([np.nan if tok == "NA" else float(tok) for tok in line.split()])
    return np.array(rows, dtype=np.float64)


def read_gene_table(path: Path):
    """Read a two-column CSV written by R's write.csv into (names, values)."""
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader)
        names, values = [], []
        for row in reader:
            names.append(row[0])
            values.append(float(row[1]))
    return np.array(names, dtype=object), np.array(values, dtype=np.float64)


class ReferenceCase:
    """One R reference case: its inputs, hypermat, and per-quadrant gene lists."""

    def __init__(self, name: str):
        self.name = name
        self.list1 = read_gene_table(DATA_DIR / f"{name}_list1.csv")
        self.list2 = read_gene_table(DATA_DIR / f"{name}_list2.csv")
        self.hypermat = read_matrix(DATA_DIR / f"{name}_hypermat.tsv")

    def genelist(self, quadrant: str):
        return {
            "list1": read_gene_list(DATA_DIR / f"{self.name}_{quadrant}_list1.txt"),
            "list2": read_gene_list(DATA_DIR / f"{self.name}_{quadrant}_list2.txt"),
            "overlap": read_gene_list(DATA_DIR / f"{self.name}_{quadrant}_overlap.txt"),
        }


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def reference():
    return ReferenceCase
