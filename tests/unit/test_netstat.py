"""Plan de Pruebas Unitarias PU-NETSTAT: Modulo netStat.py.

Evalua la inicializacion de las 4 tablas hash de red, extraccion del vector de 100
caracteristicas por paquete (MI, HH, HH_jit, HpHp) y correspondencia de encabezados.
"""

from __future__ import annotations

import numpy as np
import pytest

from netStat import netStat


@pytest.mark.unit
class TestNetStat:
    """Casos de prueba unitarios para la clase netStat."""

    def test_ut_ns_001_default_and_custom_lambdas_initialization(self) -> None:
        """UT-NS-001: Verifica inicializacion por defecto (5 lambdas -> 100 features) y lambdas personalizadas."""
        ns_default = netStat()
        assert ns_default.Lambdas == [5, 3, 1, 0.1, 0.01]
        assert ns_default.HostLimit == 255
        assert len(ns_default.getNetStatHeaders()) == 100

        ns_custom = netStat(Lambdas=[1.0, 0.1], HostLimit=50, HostSimplexLimit=100)
        assert ns_custom.Lambdas == [1.0, 0.1]
        # 20 features por cada Lambda -> 2 * 20 = 40 features
        assert len(ns_custom.getNetStatHeaders()) == 40

    def test_ut_ns_002_find_direction_ipv4_ipv6_and_l2(self) -> None:
        """UT-NS-002: Verifica extraccion de prefijos de subred en findDirection para IPv4, IPv6 y L2."""
        ns = netStat()
        src_v4, dst_v4 = ns.findDirection(
            IPtype=0,
            srcIP="192.168.1.15",
            dstIP="10.0.0.254",
            eth_src="aa:bb:cc:dd:ee:01",
            eth_dst="aa:bb:cc:dd:ee:02",
        )
        assert src_v4 == "192.168.1"
        assert dst_v4 == "10.0.0"

        src_v6, dst_v6 = ns.findDirection(
            IPtype=1,
            srcIP="fe80::1ff:fe23:4567:890a",
            dstIP="2001:db8::8a2e:370:7334",
            eth_src="aa:bb:cc:dd:ee:01",
            eth_dst="aa:bb:cc:dd:ee:02",
        )
        assert len(src_v6) == round(len("fe80::1ff:fe23:4567:890a") / 2)
        assert len(dst_v6) == round(len("2001:db8::8a2e:370:7334") / 2)

        src_l2, dst_l2 = ns.findDirection(
            IPtype=np.nan,
            srcIP="",
            dstIP="",
            eth_src="aa:bb:cc:dd:ee:01",
            eth_dst="aa:bb:cc:dd:ee:02",
        )
        assert src_l2 == "aa:bb:cc:dd:ee:01"
        assert dst_l2 == "aa:bb:cc:dd:ee:02"

    def test_ut_ns_003_update_get_stats_tcp_udp_100_dimensions(self) -> None:
        """UT-NS-003: Verifica que updateGetStats produzca un vector finito de 100 dimensiones en trafico IP/TCP."""
        ns = netStat()
        vec1 = ns.updateGetStats(
            IPtype=0,
            srcMAC="00:11:22:33:44:55",
            dstMAC="66:77:88:99:aa:bb",
            srcIP="192.168.1.10",
            srcProtocol="443",
            dstIP="192.168.1.20",
            dstProtocol="51234",
            datagramSize=120,
            timestamp=1000.0,
        )
        vec2 = ns.updateGetStats(
            IPtype=0,
            srcMAC="66:77:88:99:aa:bb",
            dstMAC="00:11:22:33:44:55",
            srcIP="192.168.1.20",
            srcProtocol="51234",
            dstIP="192.168.1.10",
            dstProtocol="443",
            datagramSize=240,
            timestamp=1000.5,
        )
        assert vec1.shape == (100,)
        assert vec2.shape == (100,)
        assert not np.any(np.isnan(vec1))
        assert not np.any(np.isnan(vec2))

    def test_ut_ns_004_update_get_stats_arp_branch_uses_mac_addresses(self) -> None:
        """UT-NS-004: Verifica la rama condicional srcProtocol == 'arp' en HT_Hp."""
        ns = netStat()
        vec_arp = ns.updateGetStats(
            IPtype=0,
            srcMAC="00:aa:bb:cc:dd:01",
            dstMAC="ff:ff:ff:ff:ff:ff",
            srcIP="192.168.1.1",
            srcProtocol="arp",
            dstIP="192.168.1.99",
            dstProtocol="arp",
            datagramSize=42,
            timestamp=2000.0,
        )
        assert vec_arp.shape == (100,)
        assert not np.any(np.isnan(vec_arp))
        # Verifica que HT_Hp haya registrado la clave con la direccion MAC y no IP+arp
        assert any(key.startswith("00:aa:bb:cc:dd:01_") for key in ns.HT_Hp.HT)

