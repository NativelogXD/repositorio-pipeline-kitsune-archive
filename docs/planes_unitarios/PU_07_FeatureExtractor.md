# Plan de Pruebas Unitarias Específico `PU-07`: Módulo `FeatureExtractor.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-07-FE`
* **Módulo Bajo Prueba:** [`FeatureExtractor.py`](../../FeatureExtractor.py) (`FE`)
* **Suite de Prueba:** [`tests/unit/test_feature_extractor.py`](../../tests/unit/test_feature_extractor.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **83%** (139/167 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
La clase `FE` transforma tramas crudas desde archivos `.tsv` o `.pcap` en llamadas normalizadas a `netStat.updateGetStats()`. El plan unitario verifica todas las ramas de clasificación de protocolo L2/L3/L4 sin depender de que Wireshark (`tshark.exe`) esté instalado en la máquina de pruebas.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-FE-001`** | `FE.__init__()` / `__prep__()` | Ruta inexistente y archivo con extensión `.txt`. | Lanza `Exception` al validar existencia y extensión del archivo. |
| **`UT-FE-002`** | `FE.get_next_vector()` (TSV) | Archivo `.tsv` de 6 paquetes: IPv4/TCP, IPv4/UDP, IPv6/TCP, ARP, ICMP y L2 MAC puro. | Extrae 6 vectores de dimensión `(100,)` sin `NaN`; en la 7ª llamada cierra `tsvinf` y retorna `[]`. |
| **`UT-FE-003`** | `FE.get_next_vector()` (`limit`) | Archivo `.tsv` con 5 paquetes y parámetro `limit=2`. | Procesa únicamente 2 paquetes y retorna `[]` en la 3ª llamada. |
| **`UT-FE-004`** | `FE.get_next_vector()` (Scapy) | Archivo `.pcap` sintético de 5 tramas (TCP, UDP, ARP, ICMP, L2 Ether) con `_get_tshark_path` mockeado a `""`. | Activa `parse_type == "scapy"`, extrae los 5 vectores `(100,)` y retorna `[]` al concluir. |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_feature_extractor.py --cov=FeatureExtractor --cov-report=term-missing
```

