"""Plan de Pruebas Unitarias PU-CORCLUST: Modulo KitNET/corClust.py.

Evalua la acumulacion incremental de residuos, calculo de matriz de distancias de
correlacion con proteccion de varianza nula y particion recursiva del dendrograma.
"""

from __future__ import annotations

import numpy as np
import pytest

from KitNET.corClust import corClust


@pytest.mark.unit
class TestCorClust:
    """Casos de prueba unitarios para clustering incremental de correlacion corClust."""

    def test_ut_cc_001_initialization_state(self) -> None:
        """UT-CC-001: Verifica estado inicial en cero para matrices y contadores de dimension n."""
        cc = corClust(n=8)
        assert cc.n == 8
        assert cc.N == 0
        assert cc.c.shape == (8,)
        assert cc.c_r.shape == (8,)
        assert cc.c_rs.shape == (8,)
        assert cc.C.shape == (8, 8)
        assert np.all(cc.C == 0.0)

    def test_ut_cc_002_incremental_update_statistics(self) -> None:
        """UT-CC-002: Verifica actualizacion incremental de media y matriz de covariacion parcial C."""
        cc = corClust(n=3)
        x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x2 = np.array([2.0, 4.0, 6.0], dtype=np.float64)

        cc.update(x1)
        assert cc.N == 1
        np.testing.assert_allclose(cc.c, x1)
        # En N=1, c_rt = x1 - x1/1 = 0
        np.testing.assert_allclose(cc.c_r, np.zeros(3))

        cc.update(x2)
        assert cc.N == 2
        np.testing.assert_allclose(cc.c, np.array([3.0, 6.0, 9.0]))
        assert np.all(np.diag(cc.C) > 0.0)
        # Simetria de la matriz de correlacion parcial C
        np.testing.assert_allclose(cc.C, cc.C.T, rtol=1e-12)

    def test_ut_cc_003_corr_dist_constant_features_zero_division_guard(self) -> None:
        """UT-CC-003: Verifica que corrDist() no genere NaN ni Inf cuando hay caracteristicas constantes."""
        cc = corClust(n=4)
        for i in range(10):
            # Las primeras dos features varian; las dos ultimas son constantes (0.0 y 5.0)
            x = np.array([float(i), float(2 * i), 0.0, 5.0], dtype=np.float64)
            cc.update(x)

        dist_matrix = cc.corrDist()
        assert dist_matrix.shape == (4, 4)
        assert not np.any(np.isnan(dist_matrix))
        assert not np.any(np.isinf(dist_matrix))
        assert np.all(dist_matrix >= 0.0)

    def test_ut_cc_004_cluster_respects_max_clust_and_partition_completeness(
        self, rng: np.random.Generator
    ) -> None:
        """UT-CC-004: Verifica que cluster(maxClust) cree grupos de tamano <= maxClust cubriendo todas las features."""
        n_features = 15
        max_clust = 4
        cc = corClust(n=n_features)

        for _ in range(40):
            cc.update(rng.normal(0.0, 1.0, size=n_features))

        feature_map = cc.cluster(maxClust=max_clust)
        assert isinstance(feature_map, list)
        assert all(1 <= len(group) <= max_clust for group in feature_map)

        flattened = [idx for group in feature_map for idx in group]
        assert len(flattened) == n_features
        assert sorted(flattened) == list(range(n_features))

    def test_ut_cc_005_cluster_boundary_clamping(self, rng: np.random.Generator) -> None:
        """UT-CC-005: Verifica limites extremos maxClust < 1 (singletons) y maxClust > n (un solo cluster)."""
        n_features = 6
        cc = corClust(n=n_features)
        for _ in range(15):
            cc.update(rng.uniform(0.0, 10.0, size=n_features))

        # Si maxClust <= 0, se ajusta a 1 -> 6 clusters de 1 elemento
        map_singletons = cc.cluster(maxClust=0)
        assert len(map_singletons) == n_features
        assert all(len(group) == 1 for group in map_singletons)

        # Si maxClust > n, se ajusta a n -> 1 solo cluster con las 6 features
        map_single_group = cc.cluster(maxClust=100)
        assert len(map_single_group) == 1
        assert sorted(map_single_group[0]) == list(range(n_features))

