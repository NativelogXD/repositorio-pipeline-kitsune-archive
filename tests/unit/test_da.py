"""Plan de Pruebas Unitarias PU-DA: Modulo KitNET/dA.py (Denoising Autoencoder).

Evalua inicializacion de hiperparametros, compresion por hiddenRatio, normalizacion
min-max online, corrupcion binomial, retropropagacion SGD y fase de ejecucion.
"""

from __future__ import annotations

import numpy as np
import pytest

from KitNET.dA import dA, dA_params


@pytest.mark.unit
class TestDenoisingAutoencoder:
    """Casos de prueba unitarios para dA_params y dA."""

    def test_ut_da_001_hidden_ratio_ceil_calculation(self) -> None:
        """UT-DA-001: Verifica que n_hidden = ceil(n_visible * hiddenRatio) e inicializacion de pesos."""
        params = dA_params(n_visible=10, n_hidden=0, lr=0.1, corruption_level=0.0, gracePeriod=100, hiddenRatio=0.75)
        ae = dA(params)
        assert ae.params.n_hidden == 8  # ceil(10 * 0.75) = 8
        assert ae.W.shape == (10, 8)
        assert ae.W_prime.shape == (8, 10)
        assert ae.hbias.shape == (8,)
        assert ae.vbias.shape == (10,)
        # Verifica pesos atados (W_prime es transpuesta compartida de W)
        np.testing.assert_allclose(ae.W_prime, ae.W.T)
        bound = 1.0 / 10.0
        assert np.all((ae.W >= -bound) & (ae.W <= bound))

    def test_ut_da_002_deterministic_seed_reproducibility(self) -> None:
        """UT-DA-002: Verifica que dos instancias de dA con mismos parametros tengan pesos identicos (semilla 1234)."""
        ae1 = dA(dA_params(n_visible=6, hiddenRatio=0.5))
        ae2 = dA(dA_params(n_visible=6, hiddenRatio=0.5))
        np.testing.assert_array_equal(ae1.W, ae2.W)

    def test_ut_da_003_online_minmax_normalization_tracking(self) -> None:
        """UT-DA-003: Verifica que train() actualice dinamicamente norm_min y norm_max por dimension."""
        ae = dA(dA_params(n_visible=3, hiddenRatio=0.75, gracePeriod=0))
        x1 = np.array([10.0, -5.0, 0.0], dtype=np.float64)
        x2 = np.array([2.0, 8.0, -3.0], dtype=np.float64)

        rmse1 = ae.train(x1)
        assert rmse1 >= 0.0
        np.testing.assert_array_equal(ae.norm_min, x1)
        np.testing.assert_array_equal(ae.norm_max, x1)

        rmse2 = ae.train(x2)
        assert rmse2 >= 0.0
        np.testing.assert_array_equal(ae.norm_min, np.array([2.0, -5.0, -3.0]))
        np.testing.assert_array_equal(ae.norm_max, np.array([10.0, 8.0, 0.0]))
        assert ae.n == 2

    def test_ut_da_004_corrupted_input_validation_and_masking(self) -> None:
        """UT-DA-004: Verifica get_corrupted_input con corruption_level valido e invalido (>=1)."""
        ae = dA(dA_params(n_visible=20, corruption_level=0.3))
        x = np.ones(20, dtype=np.float64) * 5.0
        corrupted = ae.get_corrupted_input(x, 0.3)
        assert corrupted.shape == x.shape
        assert np.all(np.isin(corrupted, [0.0, 5.0]))
        # Verificar tambien el camino train() con corruption_level > 0.0 (linea 90)
        rmse_corrupted = ae.train(x)
        assert rmse_corrupted >= 0.0

        with pytest.raises(AssertionError):
            ae.get_corrupted_input(x, 1.0)

    def test_ut_da_005_grace_period_gating_in_execute(self) -> None:
        """UT-DA-005: Verifica que execute() retorne 0.0 mientras inGrace() sea True y RMSE > 0 despues."""
        ae = dA(dA_params(n_visible=4, lr=0.1, gracePeriod=3, hiddenRatio=0.75))
        sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        assert ae.inGrace() is True
        assert ae.execute(sample) == 0.0

        for idx in range(3):
            ae.train(sample + float(idx))

        assert ae.n == 3
        assert ae.inGrace() is False
        score = float(ae.execute(sample))
        assert score > 0.0

    def test_ut_da_006_sgd_convergence_on_stationary_distribution(self, rng: np.random.Generator) -> None:
        """UT-DA-006: Verifica que el error RMSE de reconstruccion disminuya tras entrenamiento SGD en datos estacionarios."""
        ae = dA(dA_params(n_visible=5, lr=0.15, corruption_level=0.0, gracePeriod=10, hiddenRatio=0.8))
        base_pattern = np.array([0.2, 0.8, 0.4, 0.6, 0.5], dtype=np.float64)

        # Calibrar extremos min-max en las primeras dos observaciones
        ae.train(np.zeros(5, dtype=np.float64))
        ae.train(np.ones(5, dtype=np.float64))

        initial_rmse = float(ae.train(base_pattern))
        final_rmse = initial_rmse
        for _ in range(150):
            noisy = np.clip(base_pattern + rng.normal(0.0, 0.005, size=5), 0.0, 1.0)
            final_rmse = float(ae.train(noisy))

        assert final_rmse < initial_rmse

        # Un patron anomalo ortogonal debe producir mayor RMSE que el patron normal aprendido
        normal_exec_rmse = float(ae.execute(base_pattern))
        anomalous_pattern = np.array([5.0, -4.0, 6.0, -3.0, 4.5], dtype=np.float64)
        anomaly_exec_rmse = float(ae.execute(anomalous_pattern))
        assert anomaly_exec_rmse > normal_exec_rmse
