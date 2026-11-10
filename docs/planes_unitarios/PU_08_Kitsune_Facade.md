# Plan de Pruebas Unitarias Específico `PU-08`: Módulo `Kitsune.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-08-KITSUNE`
* **Módulo Bajo Prueba:** [`Kitsune.py`](../../Kitsune.py) (`Kitsune`)
* **Suite de Prueba:** [`tests/unit/test_kitsune.py`](../../tests/unit/test_kitsune.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (11/11 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
La clase `Kitsune` actúa como fachada principal del sistema, conectando la salida de `FE.get_next_vector()` con la entrada de `KitNET.process(x)` en cada llamada a `proc_next_packet()` y devolviendo `-1` cuando el flujo de paquetes llega a su fin.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-KIT-001`** | `Kitsune.__init__()` | Inicialización con archivo `.tsv`, `limit=10, max_autoencoder_size=8, FM_grace_period=5, AD_grace_period=10`. | Verifica que `FE.get_num_features() == 100` y que `AnomDetector` reciba exactamente `n=100, m=8, FM_grace=5, AD_grace=10`. |
| **`UT-KIT-002`** | `Kitsune.proc_next_packet()` | Secuencia de 15 paquetes en `.tsv` con `FM_grace=4, AD_grace=5`. | Primeros 10 paquetes retornan `0.0` (fases FM y AD); paquetes 11 a 15 retornan $\text{RMSE} > 0.0$ (fase de ejecución); llamada 16 y posteriores retornan `-1`. |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_kitsune.py --cov=Kitsune --cov-report=term-missing
```

