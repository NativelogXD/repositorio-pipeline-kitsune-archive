"""Configuracion compartida y fixtures deterministas para las pruebas unitarias de Kitsune-py."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# Asegurar que la raiz del repositorio este en sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

TSV_HEADER = (
    "frame.time_epoch\tframe.len\teth.src\teth.dst\tip.src\tip.dst\t"
    "tcp.srcport\ttcp.dstport\tudp.srcport\tudp.dstport\ticmp.type\ticmp.code\t"
    "arp.opcode\tarp.src.hw_mac\tarp.src.proto_ipv4\tarp.dst.hw_mac\t"
    "arp.dst.proto_ipv4\tipv6.src\tipv6.dst\n"
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Generador pseudoaleatorio determinista con semilla fija para reproducibilidad."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_tsv_factory(tmp_path: Path) -> Callable[[list[list[str]], str], Path]:
    """Crea un archivo TSV temporal con el esquema exacto de 19 columnas de tshark."""

    def _create_tsv(rows: list[list[str]], filename: str = "sample_traffic.tsv") -> Path:
        tsv_path = tmp_path / filename
        lines = [TSV_HEADER]
        for row in rows:
            if len(row) != 19:
                raise ValueError(f"Cada fila TSV debe tener exactamente 19 columnas, recibido: {len(row)}")
            lines.append("\t".join(row) + "\n")
        tsv_path.write_text("".join(lines), encoding="utf-8")
        logger.debug("Archivo TSV sintetico creado en %s con %d paquetes", tsv_path, len(rows))
        return tsv_path

    return _create_tsv

