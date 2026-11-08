# Plan de Pruebas Unitarias Específico `PU-03`: Módulo `KitNET/corClust.py`

## 1. Identificación y Metadatos
* **ID del Plan:** `PU-03-CORCLUST`
* **Módulo Bajo Prueba:** [`KitNET/corClust.py`](../../KitNET/corClust.py) (`corClust`)
* **Suite de Prueba:** [`tests/unit/test_corclust.py`](../../tests/unit/test_corclust.py)
* **Etapa del Pipeline CI:** `Stage 2 - Pruebas Unitarias (Pytest + Coverage)`
* **Cobertura Alcanzada:** **100%** (37/37 líneas ejecutadas)

## 2. Intuición Técnica y Alcance
La clase `corClust` implementa el *Feature Mapper* de Kitsune. Calcula de forma incremental la correlación entre las $n$ características del flujo y construye un árbol jerárquico aglomerativo que se poda recursivamente (`__breakClust__`) para garantizar que ningún autoencoder del ensamble reciba más de $m = \text{maxClust}$ entradas.

## 3. Especificación de Casos de Prueba Unitarios

| ID Caso | Método Bajo Prueba | Descripción y Datos de Entrada | Oráculo / Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **`UT-CC-001`** | `corClust.__init__(n=8)` | Inicialización con $n=8$ características. | Acumuladores `c, c_r, c_rs` de dimensión `(8,)`, matriz `C` de `(8, 8)` en cero y `N == 0`. |
| **`UT-CC-002`** | `corClust.update(x)` | Dos vectores colineales `x1=[1, 2, 3]` y `x2=[2, 4, 6]`. | Actualización exacta de suma lineal `c`, diagonal positiva en `C` y simetría $C = C^\top$. |
| **`UT-CC-003`** | `corClust.corrDist()` | 10 vectores en $\mathbb{R}^4$ donde las dimensiones 2 y 3 son constantes. | Sustitución de denominadores nulos por `1e-100`, recorte `D[D < 0] = 0` y ausencia total de `NaN`/`Inf`. |
| **`UT-CC-004`** | `corClust.cluster(maxClust)` | 40 vectores gaussianos en $\mathbb{R}^{15}$ con `maxClust=4`. | Cada sublista tiene longitud $1 \le |v_i| \le 4$ y la unión ordenada de índices es `list(range(15))`. |
| **`UT-CC-005`** | `corClust.cluster()` (Fronteras) | `n=6` con valores fuera de rango `maxClust=0` y `maxClust=100`. | Clamping inferior (`maxClust=1` $\to$ 6 clústeres unitarios) y superior (`maxClust=6` $\to$ 1 clúster completo). |

## 4. Comando de Verificación
```powershell
python -m pytest tests/unit/test_corclust.py --cov=KitNET.corClust --cov-report=term-missing
```

