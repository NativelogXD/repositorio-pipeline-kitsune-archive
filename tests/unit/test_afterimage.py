"""Plan de Pruebas Unitarias PU-AFTERIMAGE: Modulo AfterImage.py.

Evalua estadisticas incrementales amortiguadas 1D (incStat), estadisticas covariantes
2D (incStat_cov) y la base de datos en memoria de flujos (incStatDB).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from AfterImage import incStat, incStat_cov, incStatDB


@pytest.mark.unit
class TestIncStat1D:
    """Casos de prueba unitarios para estadisticas incrementales 1D (incStat)."""

    def test_ut_ai_001_exponential_decay_and_moments(self) -> None:
        """UT-AI-001: Verifica factor de decaimiento 2^(-Lambda * dt) y calculo de peso, media, var y std."""
        stat = incStat(Lambda=1.0, ID="flow_A", init_time=0.0, isTypeDiff=False)
        stat.insert(v=10.0, t=0.0)
        assert np.isclose(stat.weight(), 1.0, atol=1e-12)
        assert np.isclose(stat.mean(), 10.0, atol=1e-12)
        assert np.isclose(stat.var(), 0.0, atol=1e-9)
        assert np.isclose(stat.std(), 0.0, atol=1e-9)

        # En t=1.0 con Lambda=1.0, el factor de decaimiento es 2^(-1 * 1) = 0.5
        stat.insert(v=20.0, t=1.0)
        expected_w = 0.5 + 1.0
        expected_cf1 = 10.0 * 0.5 + 20.0
        expected_mean = expected_cf1 / expected_w
        assert np.isclose(stat.weight(), expected_w, atol=1e-12)
        assert np.isclose(stat.mean(), expected_mean, atol=1e-12)
        assert stat.var() > 0.0
        assert np.isclose(stat.std(), math.sqrt(stat.var()), atol=1e-12)

    def test_ut_ai_002_type_diff_jitter_mode(self) -> None:
        """UT-AI-002: Verifica modo isTypeDiff=True para calculo de jitter (intervalos entre llegadas)."""
        jit = incStat(Lambda=0.5, ID="jitter_AB", init_time=10.0, isTypeDiff=True)
        # Mismo timestamp o anterior -> dif <= 0 -> v = 0
        jit.insert(v=999.0, t=10.0)
        assert np.isclose(jit.mean(), 0.0, atol=1e-12)

        # Llegada en t=12.0 -> dif = 2.0
        jit.insert(v=999.0, t=12.0)
        assert jit.mean() > 0.0
        assert jit.lastTimestamp == 12.0

    def test_ut_ai_003_radius_and_magnitude_calculations(self) -> None:
        """UT-AI-003: Verifica calculo de magnitud euclidiana de medias y radio entre dos incStat."""
        s1 = incStat(Lambda=1.0, ID="s1", init_time=0.0)
        s2 = incStat(Lambda=1.0, ID="s2", init_time=0.0)
        s1.insert(v=3.0, t=0.0)
        s2.insert(v=4.0, t=0.0)

        # Magnitud de medias (3, 4) = sqrt(3^2 + 4^2) = 5.0
        assert np.isclose(s1.magnitude([s2]), 5.0, atol=1e-9)
        assert np.isclose(s1.radius([s2]), 0.0, atol=1e-9)

    def test_ut_ai_004_headers_1d_and_2d_generation(self) -> None:
        """UT-AI-004: Verifica formato de encabezados 1D y 2D con y sin sufijos."""
        s1 = incStat(Lambda=1.0, ID="host1", init_time=0.0)
        assert s1.getHeaders_1D(suffix=True) == ["weight_host1", "mean_host1", "std_host1"]
        assert s1.getHeaders_1D(suffix=False) == ["weight_0", "mean_0", "std_0"]

        hdrs_2d = s1.getHeaders_2D(ID2="host2", suffix=True)
        assert len(hdrs_2d) == 7
        assert "covariance__host1__host2" in hdrs_2d
        assert "pcc__host1__host2" in hdrs_2d


@pytest.mark.unit
class TestIncStatCov2D:
    """Casos de prueba unitarios para covarianza y correlacion de dos flujos (incStat_cov)."""

    def test_ut_ai_005_covariance_and_pcc_tracking(self) -> None:
        """UT-AI-005: Verifica actualizacion acoplada de residuos CF3, covarianza, PCC y get_stats1..4."""
        s1 = incStat(Lambda=0.1, ID="A", init_time=0.0)
        s2 = incStat(Lambda=0.1, ID="B", init_time=0.0)
        cov_tracker = incStat_cov(s1, s2, init_time=0.0)
        s1.covs.append(cov_tracker)
        s2.covs.append(cov_tracker)

        for step in range(1, 8):
            val = float(step * 10)
            s1.insert(v=val, t=float(step))
            s2.insert(v=val * 2.0, t=float(step) + 0.1)

        stats1 = cov_tracker.get_stats1()
        stats2 = cov_tracker.get_stats2()
        stats3 = cov_tracker.get_stats3()
        stats4 = cov_tracker.get_stats4()

        assert len(stats1) == 2
        assert len(stats2) == 4
        assert len(stats3) == 8
        assert len(stats4) == 10
        assert not np.any(np.isnan(stats4))

        # Verificar delegacion desde incStat hacia incStat_cov: cov(), pcc(), cov_pcc(), allstats_2D()
        assert np.isclose(s1.cov("B"), cov_tracker.cov(), rtol=1e-12)
        assert np.isclose(s1.pcc("B"), cov_tracker.pcc(), rtol=1e-12)
        assert s1.cov_pcc("B") == stats1
        assert len(s1.allstats_2D("B")) == 7
        assert np.isnan(s1.cov("UNKNOWN")[0])
        assert np.isnan(s1.pcc("UNKNOWN")[0])
        assert all(np.isnan(val) for val in s1.cov_pcc("UNKNOWN"))

        for ver, expected_len in [(1, 2), (2, 4), (3, 8), (4, 5), (5, 10)]:
            assert len(cov_tracker.getHeaders(ver=ver, suffix=True)) == expected_len


@pytest.mark.unit
class TestIncStatDB:
    """Casos de prueba unitarios para el contenedor hash de flujos incStatDB."""

    def test_ut_ai_006_register_limit_raises_lookup_error(self) -> None:
        """UT-AI-006: Verifica que superar el limite de entradas en incStatDB lance LookupError."""
        db = incStatDB(limit=2)
        db.register("host_1", Lambda=1.0)
        db.register("host_2", Lambda=1.0)
        # Re-registrar una clave existente no debe lanzar error
        db.register("host_1", Lambda=1.0)

        with pytest.raises(LookupError, match="would exceed incStatHT 1D limit"):
            db.register("host_3", Lambda=1.0)

    def test_ut_ai_007_missing_stream_queries_return_nan(self) -> None:
        """UT-AI-007: Verifica que consultar flujos inexistentes en get_1D_Stats y get_2D_Stats retorne NaN."""
        db = incStatDB(limit=100)
        res_1d = db.get_1D_Stats("non_existent", Lambda=1.0)
        assert len(res_1d) == 3
        assert np.all(np.isnan(res_1d))

        res_2d = db.get_2D_Stats("non_existent_1", "non_existent_2", Lambda=1.0)
        assert len(res_2d) == 2
        assert np.all(np.isnan(res_2d))

        stats_all, ids_all = db.get_all_2D_Stats("non_existent", Lambda=1.0)
        assert stats_all == []
        assert ids_all == []

    def test_ut_ai_008_update_get_1d2d_and_clean_old_records(self) -> None:
        """UT-AI-008: Verifica update_get_1D2D_Stats (7 dims), get_nD_Stats y poda cleanOutOldRecords."""
        db = incStatDB(limit=100, default_lambda=1.0)
        vec_7d = db.update_get_1D2D_Stats("10.0.0.1", "10.0.0.2", t1=1.0, v1=64.0, Lambda=1.0)
        assert len(vec_7d) == 7
        assert not np.any(np.isnan(vec_7d))

        nd_stats = db.get_nD_Stats(["10.0.0.1", "10.0.0.2"], Lambda=1.0)
        assert len(nd_stats) == 2

        # Consultas directas sobre flujos ya registrados
        existing_1d = db.get_1D_Stats("10.0.0.1", Lambda=1.0)
        assert len(existing_1d) == 3 and not np.any(np.isnan(existing_1d))
        existing_2d = db.get_2D_Stats("10.0.0.1", "10.0.0.2", Lambda=1.0)
        assert len(existing_2d) == 2 and not np.any(np.isnan(existing_2d))
        all_2d_stats, all_2d_ids = db.get_all_2D_Stats("10.0.0.1", Lambda=1.0)
        assert len(all_2d_stats) == 1 and all_2d_ids == [["10.0.0.1", "10.0.0.2"]]
        level1_2d = db.update_get_2D_Stats("10.0.0.1", "10.0.0.2", t1=1.5, v1=128.0, Lambda=1.0, level=1)
        assert len(level1_2d) == 2
        hdrs_nd = db.getHeaders_nD(Lambda=1.0, IDs=["10.0.0.1", "10.0.0.2"])
        assert len(hdrs_nd) == 2

        # En t=20.0 con Lambda=1.0, el peso decae por 2^(-19) << 0.01 -> ambos registros deben limpiarse
        removed = db.cleanOutOldRecords(cutoffWeight=0.01, curTime=20.0)
        assert removed == 2
        assert len(db.HT) == 0
