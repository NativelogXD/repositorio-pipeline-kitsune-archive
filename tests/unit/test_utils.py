"""Plan de Pruebas Unitarias PU-UTILS: Modulo KitNET/utils.py.

Evalua matematicamente las funciones de activacion, derivadas, funciones de densidad
de probabilidad normal, CDF logaritmica inversa y media movil circular (rollmean).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from KitNET import utils


@pytest.mark.unit
class TestUtilsDistributions:
    """Casos de prueba unitarios para funciones estadisticas (pdf e invLogCDF)."""

    def test_ut_utils_001_pdf_standard_normal(self) -> None:
        """UT-UTILS-001: Verifica que pdf(0, 0, 1) coincida con 1/sqrt(2*pi)."""
        result = float(utils.pdf(0.0, 0.0, 1.0))
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        assert np.isclose(result, expected, rtol=1e-7)

    def test_ut_utils_002_pdf_vectorized_symmetry(self) -> None:
        """UT-UTILS-002: Verifica simetria gaussiana vectorizada alrededor de mu."""
        mu = 5.0
        sigma = 2.0
        offsets = np.array([0.5, 1.0, 2.5, 4.0], dtype=np.float64)
        left = utils.pdf(mu - offsets, mu, sigma)
        right = utils.pdf(mu + offsets, mu, sigma)
        np.testing.assert_allclose(left, right, rtol=1e-12)

    def test_ut_utils_003_inv_log_cdf_matches_scipy_logsf(self) -> None:
        """UT-UTILS-003: Verifica que invLogCDF equivalga a log(1 - CDF) = norm.logcdf(-z)."""
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        mu = 1.0
        sigma = 0.5
        result = utils.invLogCDF(x, mu, sigma)
        expected = norm.logcdf(-(x - mu) / sigma)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.unit
class TestUtilsActivations:
    """Casos de prueba unitarios para funciones de activacion neuronal y derivadas."""

    def test_ut_utils_004_sigmoid_bounds_and_origin(self) -> None:
        """UT-UTILS-004: Verifica sigmoid(0)=0.5 y acotamiento estricto en [0, 1]."""
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float64)
        y = utils.sigmoid(x)
        assert np.isclose(y[2], 0.5, atol=1e-12)
        assert np.all((y > 0.0) & (y < 1.0))
        assert np.all(np.diff(y) > 0.0)

    def test_ut_utils_005_dsigmoid_identity(self) -> None:
        """UT-UTILS-005: Verifica dsigmoid(y) = y * (1 - y) para salidas activadas."""
        y = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)
        grad = utils.dsigmoid(y)
        expected = y * (1.0 - y)
        np.testing.assert_allclose(grad, expected, rtol=1e-12)
        assert np.isclose(utils.dsigmoid(0.5), 0.25, atol=1e-12)

    def test_ut_utils_006_tanh_and_dtanh(self) -> None:
        """UT-UTILS-006: Verifica tanh(0)=0, imparidad y dtanh(y) = 1 - y^2."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        y = utils.tanh(x)
        np.testing.assert_allclose(y, -utils.tanh(-x), rtol=1e-12)
        np.testing.assert_allclose(utils.dtanh(y), 1.0 - y**2, rtol=1e-12)

    def test_ut_utils_007_softmax_1d_and_2d_simplex(self) -> None:
        """UT-UTILS-007: Verifica que softmax sume 1.0 en tensores 1D y 2D con invarianza al desplazamiento."""
        vec_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out_1d = utils.softmax(vec_1d)
        out_1d_shifted = utils.softmax(vec_1d + 1000.0)
        assert np.isclose(np.sum(out_1d), 1.0, atol=1e-12)
        np.testing.assert_allclose(out_1d, out_1d_shifted, rtol=1e-12)

        mat_2d = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float64)
        out_2d = utils.softmax(mat_2d)
        assert out_2d.shape == (2, 3)
        np.testing.assert_allclose(np.sum(out_2d, axis=1), np.ones(2), atol=1e-12)

    def test_ut_utils_008_relu_and_drelu(self) -> None:
        """UT-UTILS-008: Verifica ReLU(x)=max(0,x) y su subgradiente dReLU(x)."""
        x = np.array([-5.0, -0.01, 0.0, 0.01, 5.0], dtype=np.float64)
        np.testing.assert_allclose(utils.ReLU(x), np.array([0.0, 0.0, 0.0, 0.01, 5.0]))
        np.testing.assert_allclose(utils.dReLU(x), np.array([0.0, 0.0, 0.0, 1.0, 1.0]))


@pytest.mark.unit
class TestUtilsRollmean:
    """Casos de prueba unitarios para el buffer circular rollmean."""

    def test_ut_utils_009_rollmean_circular_window(self) -> None:
        """UT-UTILS-009: Verifica actualizacion circular de ventana k=3 y reemplazo FIFO."""
        rm = utils.rollmean(k=3)
        assert rm.winsize == 3
        assert rm.pointer == 0

        # Ventana inicial: [0, 0, 0] -> tras insertar 3: [3, 0, 0] -> media 1.0
        assert np.isclose(rm.apply(3.0), 1.0)
        # Tras insertar 6: [3, 6, 0] -> media 3.0
        assert np.isclose(rm.apply(6.0), 3.0)
        # Tras insertar 9: [3, 6, 9] -> media 6.0 y puntero envuelve a 0
        assert np.isclose(rm.apply(9.0), 6.0)
        assert rm.pointer == 0
        # Tras insertar 12: reemplaza 3 -> [12, 6, 9] -> media 9.0
        assert np.isclose(rm.apply(12.0), 9.0)
        assert rm.pointer == 1

