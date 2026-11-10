# Plan de Pruebas Unitarias Específico `PU-05`: Módulo `AfterImage.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-05-AFTERIMAGE`
* **Módulo Bajo Prueba:** [`AfterImage.py`](../../AfterImage.py) (`incStat`, `incStat_cov`, `incStatDB`)
* **Suite de Prueba:** [`tests/unit/test_afterimage.py`](../../tests/unit/test_afterimage.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **98%** (284/291 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
`AfterImage` mantiene perfiles estadísticos de flujos de red en tiempo $O(1)$ usando decaimiento exponencial $2^{-\lambda \Delta t}$. El plan verifica:
1. Cálculo exacto de peso $w$, media $\mu$, varianza $\sigma^2$, desviación estándar $\sigma$, radio y magnitud en `incStat`.
2. Modo diferencial `isTypeDiff=True` para medir intervalos entre llegadas (*jitter*).
3. Covarianza incremental de residuos $CF_3$ y coeficiente de correlación de Pearson en `incStat_cov`.
4. Registro acotado (`LookupError`), consultas 1D/2D/nD para flujos existentes e inexistentes, y recolección de basura por peso mínimo (`cleanOutOldRecords`) en `incStatDB`.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Clase / Método | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-AI-001`** | `incStat.insert()`, `processDecay()` | `Lambda=1.0`, inserciones `(10.0, t=0.0)` y `(20.0, t=1.0)`. | Factor $\gamma = 0.5$, peso $w = 1.5$, media $\mu = 25/1.5 \approx 16.6667$, $\sigma = \sqrt{\text{var}}$. |
| **`UT-AI-002`** | `incStat` (`isTypeDiff=True`) | `init_time=10.0`, inserciones en $t=10.0$ y $t=12.0$. | Registra $\Delta t = 0.0$ en la primera inserción y $\Delta t = 2.0$ en la segunda. |
| **`UT-AI-003`** | `incStat.magnitude()`, `radius()` | Dos flujos con medias `3.0` y `4.0` en $t=0$. | Magnitud $\sqrt{3^2 + 4^2} = 5.0$ y radio $0.0$. |
| **`UT-AI-004`** | `getHeaders_1D()`, `getHeaders_2D()` | Generación de etiquetas para `host1` y `host2`. | 3 etiquetas 1D y 7 etiquetas combinadas 1D+2D con prefijos esperados. |
| **`UT-AI-005`** | `incStat_cov` | 7 actualizaciones intercaladas entre flujos `A` y `B`. | Vectores `get_stats1..4` de longitudes `2, 4, 8, 10` sin `NaN` y encabezados `ver=1..5`. |
| **`UT-AI-006`** | `incStatDB.register()` | `limit=2`, registro de 3 flujos distintos. | Lanza `LookupError` al intentar insertar el tercer flujo distinto. |
| **`UT-AI-007`** | `incStatDB.get_1D/2D_Stats()` | Consultas sobre claves no registradas. | Retorna `[np.nan]*3`, `[np.nan]*2` y `([], [])`. |
| **`UT-AI-008`** | `update_get_1D2D_Stats()`, `cleanOutOldRecords()` | Registro de 2 flujos en $t=1.0$ y poda en $t=20.0$ con `cutoffWeight=0.01`. | Retorna vector 7D válido, resuelve consultas 1D/2D/nD y elimina los 2 registros caducados. |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_afterimage.py --cov=AfterImage --cov-report=term-missing
```
