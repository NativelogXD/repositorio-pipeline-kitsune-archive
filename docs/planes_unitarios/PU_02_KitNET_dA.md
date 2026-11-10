# Plan de Pruebas Unitarias Específico `PU-02`: Módulo `KitNET/dA.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-02-DA`
* **Módulo Bajo Prueba:** [`KitNET/dA.py`](../../KitNET/dA.py) (`dA_params` y `dA`)
* **Suite de Prueba:** [`tests/unit/test_da.py`](../../tests/unit/test_da.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (65/65 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
La clase `dA` implementa un *Denoising Autoencoder* de tres capas con pesos atados ($W' = W^\top$) que opera de forma estrictamente *online* (actualizando pesos y cotas de normalización muestra a muestra en $O(d_v \cdot d_h)$). Este plan verifica:
1. El dimensionamiento automático de la capa oculta $d_h = \lceil d_v \cdot \text{hiddenRatio} \rceil$.
2. El determinismo de la inicialización uniforme en $[-1/d_v, 1/d_v]$ mediante `RandomState(1234)`.
3. El seguimiento incremental de extremos `norm_min` y `norm_max` para escalar en $[0, 1]$.
4. El enmascaramiento binomial de entradas (`get_corrupted_input`).
5. El descenso del error RMSE de reconstrucción bajo entrenamiento SGD y la sensibilidad ante anomalías en `execute()`.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-DA-001`** | `dA.__init__()` | `n_visible=10, hiddenRatio=0.75`. | `n_hidden == 8`, `W.shape == (10, 8)`, `W_prime` es vista transpuesta `W.T` y $|W_{ij}| \le 0.1$. |
| **`UT-DA-002`** | `dA.__init__()` | Dos instancias independientes con `n_visible=6, hiddenRatio=0.5`. | Matrices de pesos `W` idénticas bit a bit por semilla fija `1234`. |
| **`UT-DA-003`** | `dA.train(x)` | Vectores `x1=[10, -5, 0]` y `x2=[2, 8, -3]`. | `norm_min == [2, -5, -3]`, `norm_max == [10, 8, 0]`, `n == 2` y RMSE finito $\ge 0$. |
| **`UT-DA-004`** | `get_corrupted_input()`, `train()` | Vector constante `5.0` de dimensión 20 con `corruption_level=0.3` y `1.0`. | Con `0.3` produce valores en `{0.0, 5.0}` y entrena sin error; con `1.0` lanza `AssertionError`. |
| **`UT-DA-005`** | `execute(x)`, `inGrace()` | `gracePeriod=3`; evaluación antes y después de 3 llamadas a `train()`. | Mientras `n < 3`: `inGrace() == True` y `execute(x) == 0.0`. Con `n == 3`: `inGrace() == False` y `execute(x) > 0.0`. |
| **`UT-DA-006`** | `train(x)`, `execute(x)` | 150 iteraciones sobre distribución estacionaria $\mathcal{N}(\mu_0, 0.005^2)$ en $\mathbb{R}^5$ y evaluación de vector anómalo. | Convergencia $\text{RMSE}_{\text{final}} < \text{RMSE}_{\text{inicial}}$ y discriminación $\text{RMSE}_{\text{anomalía}} > \text{RMSE}_{\text{normal}}$. |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_da.py --cov=KitNET.dA --cov-report=term-missing
```

