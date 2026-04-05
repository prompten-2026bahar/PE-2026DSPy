# /translation/knn.md

## dspy.KNN

```python
dspy.KNN(k: int, trainset: list[Example], vectorizer: Embedder)
```

Bir eğitim setinden (training set) benzer örnekleri bulan bir k-en yakın komşu (k-nearest neighbors) getiricisidir (retriever).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `k` | `int` | Aranıp getirilecek (retrieve) en yakın komşuların sayısı. | **Gerekli (required)** |
| `trainset` | `list[Example]` | İçinde arama yapılacak eğitim örneklerinin listesi. | **Gerekli (required)** |
| `vectorizer` | `Embedder` | Vektörleştirme (vectorization) için kullanılacak `Embedder`. | **Gerekli (required)** |

**Örnekler (Examples):**

```python
import dspy
from sentence_transformers import SentenceTransformer

# Örneklerle bir eğitim veri seti oluşturun
trainset = [
    dspy.Example(input="hello", output="world"),
    # ... daha fazla örnek ...
]

# KNN'i bir sentence transformer modeliyle başlatın
knn = KNN(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Benzer örnekleri bulun
similar_examples = knn(input="hello")
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/knn.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(**kwargs) -> list
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/predict/knn.py`*
