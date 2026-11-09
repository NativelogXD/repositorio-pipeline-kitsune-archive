"""Plan de Pruebas Unitarias PU-KITSUNE: Modulo Kitsune.py.

Evalua la fachada principal Kitsune integrando el flujo de proc_next_packet() a traves
de las fases de mapeo (FM), entrenamiento (AD) y ejecucion, hasta el centinela EOF (-1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from Kitsune import Kitsune


@pytest.mark.unit
class TestKitsuneFacade:
    """Casos de prueba unitarios para la fachada principal Kitsune."""

    def test_ut_kit_001_initialization_wires_fe_and_kitnet(
        self, synthetic_tsv_factory: Callable[[list[list[str]], str], Path]
    ) -> None:
        """UT-KIT-001: Verifica que Kitsune inicialice FE con 100 features y configure KitNET."""
        row = [
            "1000.0", "64", "00:11:22:33:44:55", "66:77:88:99:aa:bb",
            "192.168.1.1", "192.168.1.2", "443", "50000", "", "",
            "", "", "", "", "", "", "", "", "",
        ]
        tsv_path = synthetic_tsv_factory([row], "init_check.tsv")
        kitsune = Kitsune(
            file_path=str(tsv_path),
            limit=10,
            max_autoencoder_size=8,
            FM_grace_period=5,
            AD_grace_period=10,
            learning_rate=0.05,
            hidden_ratio=0.75,
        )

        assert kitsune.FE.get_num_features() == 100
        assert kitsune.AnomDetector.n == 100
        assert kitsune.AnomDetector.m == 8
        assert kitsune.AnomDetector.FM_grace_period == 5
        assert kitsune.AnomDetector.AD_grace_period == 10

    def test_ut_kit_002_proc_next_packet_lifecycle_and_eof_sentinel(
        self, synthetic_tsv_factory: Callable[[list[list[str]], str], Path]
    ) -> None:
        """UT-KIT-002: Verifica proc_next_packet() en fases de entrenamiento (0.0), ejecucion (>0) y EOF (-1)."""
        rows: list[list[str]] = []
        for idx in range(15):
            rows.append([
                f"{1000.0 + idx * 0.2:.1f}",
                str(60 + (idx % 5) * 10),
                "00:11:22:33:44:55",
                "66:77:88:99:aa:bb",
                "192.168.1.10",
                "192.168.1.20",
                str(40000 + (idx % 3)),
                "443",
                "", "", "", "", "", "", "", "", "", "", "",
            ])

        tsv_path = synthetic_tsv_factory(rows, "lifecycle.tsv")
        fm_grace = 4
        ad_grace = 5
        kitsune = Kitsune(
            file_path=str(tsv_path),
            limit=15,
            max_autoencoder_size=10,
            FM_grace_period=fm_grace,
            AD_grace_period=ad_grace,
        )

        scores: list[float] = []
        while True:
            rmse = kitsune.proc_next_packet()
            if rmse == -1:
                break
            scores.append(float(rmse))

        assert len(scores) == 15
        # Los primeros fm_grace + ad_grace + 1 = 10 paquetes corresponden a entrenamiento -> score == 0.0
        assert all(score == 0.0 for score in scores[: fm_grace + ad_grace + 1])
        # Los ultimos 5 paquetes estan en fase de ejecucion -> RMSE > 0.0
        assert all(score > 0.0 and not np.isnan(score) for score in scores[fm_grace + ad_grace + 1 :])
        # Llamada adicional tras agotar paquetes sigue devolviendo -1
        assert kitsune.proc_next_packet() == -1

    def test_ut_kit_003_custom_hyperparameters_propagation(
        self, synthetic_tsv_factory: Callable[[list[list[str]], str], Path]
    ) -> None:
        """UT-KIT-003: Verifica la propagacion correcta de hiperparametros lr y hidden_ratio al ensamble."""
        rows: list[list[str]] = []
        for idx in range(8):
            rows.append([
                f"{2000.0 + idx * 0.1:.1f}",
                str(64 + (idx % 4) * 8),
                "00:aa:bb:cc:dd:ee",
                "11:22:33:44:55:66",
                "10.0.0.1",
                "10.0.0.2",
                str(50000 + idx),
                "80",
                "", "", "", "", "", "", "", "", "", "", "",
            ])

        tsv_path = synthetic_tsv_factory(rows, "hyperparams_check.tsv")
        custom_lr = 0.02
        custom_hr = 0.5
        kitsune = Kitsune(
            file_path=str(tsv_path),
            limit=8,
            max_autoencoder_size=5,
            FM_grace_period=2,
            AD_grace_period=3,
            learning_rate=custom_lr,
            hidden_ratio=custom_hr,
        )

        assert kitsune.AnomDetector.lr == custom_lr
        assert kitsune.AnomDetector.hr == custom_hr

        # Procesar suficientes paquetes para que termine FM y se construyan los dA
        for _ in range(5):
            kitsune.proc_next_packet()

        # Validar que los autoencoders instanciados en el ensamble hayan recibido los hiperparametros
        assert len(kitsune.AnomDetector.ensembleLayer) > 0
        for ae in kitsune.AnomDetector.ensembleLayer:
            assert ae.params.lr == custom_lr
            assert ae.params.hiddenRatio == custom_hr
        assert kitsune.AnomDetector.outputLayer.params.lr == custom_lr
        assert kitsune.AnomDetector.outputLayer.params.hiddenRatio == custom_hr

    def test_ut_kit_004_empty_packet_stream_immediate_eof(
        self, synthetic_tsv_factory: Callable[[list[list[str]], str], Path]
    ) -> None:
        """UT-KIT-004: Verifica la resiliencia y retorno seguro de EOF (-1) ante un flujo sin paquetes."""
        tsv_path = synthetic_tsv_factory([], "empty_stream.tsv")
        kitsune = Kitsune(
            file_path=str(tsv_path),
            limit=10,
            max_autoencoder_size=10,
            FM_grace_period=5,
            AD_grace_period=5,
        )

        # Ante stream vacio, la primera llamada debe devolver centinela -1
        assert kitsune.proc_next_packet() == -1
        # Llamadas consecutivas deben mantenerse estables en -1 sin lanzar excepciones
        assert kitsune.proc_next_packet() == -1


