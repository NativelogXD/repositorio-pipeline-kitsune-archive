# Plan de Pruebas Unitarias Específico `PU-04`: Módulo `KitNET/KitNET.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-04-KITNET`
* **Módulo Bajo Prueba:** [`KitNET/KitNET.py`](../../KitNET/KitNET.py) (`KitNET`)
* **Suite de Prueba:** [`tests/unit/test_kitnet.py`](../../tests/unit/test_kitnet.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (61/61 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
`KitNET` coordina la transición entre las tres fases operativas del detector de anomalías:
1. **Fase 1 (`FM train-mode`):** Recolecta estadísticas de correlación en `self.FM` durante `0 <= n_trained <= FM_grace_period`. En el instante exacto `n_trained == FM_grace_period`, ejecuta `self.FM.cluster(self.m)` e instancia `ensembleLayer` y `outputLayer` mediante `__createAD__()`.
2. **Fase 2 (`AD train-mode`):** Entrena cada autoencoder de `ensembleLayer` sobre su subvector $x[v_a]$ y entrena `outputLayer` con el vector de errores $S_{l1}$, retornando `0.0` en `process(x)`.
3. **Fase 3 (`AD execute-mode`):** Cuando `n_trained > FM_grace_period + AD_grace_period`, evalúa `execute(x)` y retorna el puntaje RMSE de anomalía.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-KN-001`** | `KitNET.__init__()` | `n=10, max_autoencoder_size=0, FM_grace_period=None, AD_grace_period=500`. | `FM_grace_period` toma el valor `500`, `m` se corrige a `1` y `v` es `None`. |
| **`UT-KN-002`** | `KitNET.execute(x)` | Invocación de `execute(x)` antes de contar con `feature_map`. | Lanza `RuntimeError` con mensaje descriptivo. |
| **`UT-KN-003`** | `KitNET.__init__(feature_map=...)` | Mapeo explícito `[[0,1,2],[3,4],[5]]`, `FM_grace=0, AD_grace=5`. | Construye inmediatamente 3 autoencoders en `ensembleLayer` y 1 en `outputLayer`; tras 6 pasos de entrenamiento emite $\text{RMSE} > 0$. |
| **`UT-KN-004`** | `KitNET.process(x)` | Flujo completo con `n=12, m=4, FM_grace=10, AD_grace=15`. | Verifica transición exacta en el paquete 11 (creación del mapeo `v` y AD) y en el paquete 27 (primera ejecución con $\text{RMSE} > 0$). |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_kitnet.py --cov=KitNET.KitNET --cov-report=term-missing
```

