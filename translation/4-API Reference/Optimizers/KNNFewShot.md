# /translation/knn_fewshot.md

## dspy.KNNFewShot

```python
dspy.KNNFewShot(k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any])
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

KNNFewShot, test zamanında (test time) bir eğitim setindeki (trainset) en yakın `k` komşuyu bulmak için bellek içi (in-memory) bir KNN getiricisi (retriever) kullanan bir optimize edicidir. Bir `forward` (ileri) çağrısındaki her bir girdi örneği için, eğitim setindeki en benzer `k` örneği tanımlar ve bunları öğrenci (student) modülüne gösterim/demo (demonstrations) olarak ekler.



**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `k` | `int` | Öğrenci modeline eklenecek en yakın komşuların sayısı. | **Gerekli (required)** |
| `trainset` | `list[Example]` | Az atışlı istemleme (few-shot prompting) için kullanılacak eğitim seti. | **Gerekli (required)** |
| `vectorizer` | `Embedder` | Vektörleştirme (vectorization) için kullanılacak `Embedder`. | **Gerekli (required)** |
| `**few_shot_bootstrap_args` | `dict[str, Any]` | `BootstrapFewShot` optimize edicisi için ek argümanlar. | `{}` |

**Örnekler (Examples):**

```python
import dspy
from sentence_transformers import SentenceTransformer

# Düşünce zinciri (chain of thought) içeren bir QA (Soru-Cevap) modülü tanımlayın
qa = dspy.ChainOfThought("question -> answer")

# Örneklerle bir eğitim veri seti oluşturun
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    # ... daha fazla örnek ...
]

# KNNFewShot'ı bir sentence transformer modeliyle başlatın
knn_few_shot = KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# QA modülünü az atışlı (few-shot) öğrenme ile derleyin (compile)
compiled_qa = knn_few_shot.compile(qa)

# Derlenmiş modülü kullanın
result = compiled_qa("What is the capital of Belgium?")
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/knn_fewshot.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student, *, teacher=None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/knn_fewshot.py`*

### `get_params`

```python
get_params() -> dict[str, Any]
```

Teleprompter'ın parametrelerini alır.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `dict[str, Any]` | Teleprompter'ın parametreleri. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/teleprompt.py`*
