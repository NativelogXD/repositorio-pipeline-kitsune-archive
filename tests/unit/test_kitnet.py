"""Plan de Pruebas Unitarias PU-KITNET: Modulo KitNET/KitNET.py.

Evalua la maquina de estados del ensamble (FM train-mode -> AD train-mode -> AD execute-mode),
mapeo de caracteristicas provisto vs aprendido y propagacion de errores hacia la capa de salida.
"""

from __future__ import annotations

import numpy as np
import pytest

from KitNET.KitNET import KitNET


@pytest.mark.unit
class TestKitNET:
    """Casos de prueba unitarios para la clase orquestadora KitNET."""

    def test_ut_kn_001_init_defaults_and_grace_period_inheritance(self) -> None:
        """UT-KN-001: Verifica que FM_grace_period herede AD_grace_period cuando es None y m<=0 se ajuste a 1."""
        kn = KitNET(n=10, max_autoencoder_size=0, FM_grace_period=None, AD_grace_period=500)
        assert kn.FM_grace_period == 500
        assert kn.AD_grace_period == 500
        assert kn.m == 1
        assert kn.v is None
        assert kn.outputLayer is None
        assert kn.ensembleLayer == []

    def test_ut_kn_002_execute_without_feature_map_raises_runtime_error(self) -> None:
        """UT-KN-002: Verifica que execute() lance RuntimeError si aun no existe un feature_map."""
        kn = KitNET(n=6, max_autoencoder_size=3, FM_grace_period=10, AD_grace_period=20)
        x = np.ones(6, dtype=np.float64)
        with pytest.raises(RuntimeError, match="feature mapping has not yet been learned"):
            kn.execute(x)

    def test_ut_kn_003_predefined_feature_map_skips_fm_phase(self, rng: np.random.Generator) -> None:
        """UT-KN-003: Verifica que al suministrar feature_map se instancie inmediatamente la red de autoencoders."""
        custom_map = [[0, 1, 2], [3, 4], [5]]
        kn = KitNET(
            n=6,
            max_autoencoder_size=3,
            FM_grace_period=0,
            AD_grace_period=5,
            feature_map=custom_map,
        )
        assert kn.v == custom_map
        assert len(kn.ensembleLayer) == 3
        assert kn.outputLayer is not None
        assert kn.outputLayer.params.n_visible == 3

        # Entrenar durante AD_grace_period (n_trained: 0..5 -> 6 instancias)
        for _ in range(6):
            score = kn.process(rng.normal(0.0, 1.0, size=6))
            assert score == 0.0

        # A partir de n_trained > FM_grace + AD_grace (6 > 5), pasa a modo ejecucion
        exec_score = float(kn.process(rng.normal(0.0, 1.0, size=6)))
        assert exec_score > 0.0
        assert kn.n_executed == 1

    def test_ut_kn_004_full_lifecycle_fm_to_ad_to_execute(self, rng: np.random.Generator) -> None:
        """UT-KN-004: Verifica transicion completa FM train -> AD train -> AD execute mediante process(x)."""
        n_features = 12
        fm_grace = 10
        ad_grace = 15
        kn = KitNET(
            n=n_features,
            max_autoencoder_size=4,
            FM_grace_period=fm_grace,
            AD_grace_period=ad_grace,
            learning_rate=0.1,
            hidden_ratio=0.75,
        )

        # Fase 1: FM learning (n_trained: 0 .. fm_grace inclusive -> fm_grace + 1 paquetes)
        for step in range(fm_grace):
            out = kn.process(rng.uniform(1.0, 10.0, size=n_features))
            assert out == 0.0
            assert kn.v is None, f"El mapeo no debe existir antes del paso {fm_grace} (actual: {step})"

        # Paso exacto en que n_trained == fm_grace: se ejecuta cluster() y __createAD__()
        out_fm_done = kn.process(rng.uniform(1.0, 10.0, size=n_features))
        assert out_fm_done == 0.0
        assert kn.v is not None
        assert all(len(cluster) <= 4 for cluster in kn.v)
        assert len(kn.ensembleLayer) == len(kn.v)
        assert kn.outputLayer is not None

        # Fase 2: AD training (n_trained: fm_grace + 1 .. fm_grace + ad_grace -> ad_grace paquetes)
        for _ in range(ad_grace):
            out_ad = kn.process(rng.uniform(1.0, 10.0, size=n_features))
            assert out_ad == 0.0

        assert kn.n_trained == fm_grace + ad_grace + 1
        assert kn.n_executed == 0

        # Fase 3: AD execution (n_trained > fm_grace + ad_grace)
        rmse_score = float(kn.process(rng.uniform(1.0, 10.0, size=n_features)))
        assert rmse_score > 0.0
        assert not np.isnan(rmse_score)
        assert kn.n_executed == 1

