# Plan de Pruebas Unitarias Específico `PU-06`: Módulo `netStat.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-06-NETSTAT`
* **Módulo Bajo Prueba:** [`netStat.py`](../../netStat.py) (`netStat`)
* **Suite de Prueba:** [`tests/unit/test_netstat.py`](../../tests/unit/test_netstat.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (59/59 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
`netStat` coordina las cuatro vistas contextuales de cada paquete (`HT_MI`, `HT_H`, `HT_jit`, `HT_Hp`) a través de las 5 escalas temporales $\Lambda = [5, 3, 1, 0.1, 0.01]$, produciendo el vector de características $x \in \mathbb{R}^{100}$.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-NS-001`** | `netStat.__init__()`, `getNetStatHeaders()` | Inicialización con `Lambdas=np.nan` (default) y con `Lambdas=[1.0, 0.1]`. | Genera exactamente 100 encabezados con 5 lambdas y 40 encabezados con 2 lambdas. |
| **`UT-NS-002`** | `netStat.findDirection()` | Pruebas para `IPtype=0` (IPv4), `IPtype=1` (IPv6) e `IPtype=np.nan` (L2 MAC). | Extrae subred `/24` en IPv4, prefijo medio en IPv6 y direcciones MAC en L2. |
| **`UT-NS-003`** | `netStat.updateGetStats()` (TCP/UDP) | Tráfico bidireccional entre `192.168.1.10:443` y `192.168.1.20:51234`. | Retorna `np.ndarray` de forma `(100,)` sin `NaN` ni `Inf`. |
| **`UT-NS-004`** | `netStat.updateGetStats()` (ARP) | Paquete con `srcProtocol="arp"` y `dstProtocol="arp"`. | Ejecuta la rama condicional ARP indexando `HT_Hp` mediante `srcMAC` y `dstMAC`, retornando vector `(100,)`. |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_netstat.py --cov=netStat --cov-report=term-missing
```

