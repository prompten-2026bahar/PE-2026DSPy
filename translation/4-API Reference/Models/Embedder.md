# /translation/embedder.md

## dspy.Embedder

```python
dspy.Embedder(model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any])
```

DSPy embedding (gömme/vektörel temsil) sınıfı.

Metin girdileri için embedding'leri hesaplamak amacıyla kullanılan sınıftır. Bu sınıf, aşağıdakilerin her ikisi için de birleştirilmiş (unified) bir arayüz sağlar:

* `litellm` entegrasyonu aracılığıyla barındırılan (hosted) embedding modelleri (örn. OpenAI'nin `text-embedding-3-small` modeli).
* Sizin sağladığınız özel (custom) embedding fonksiyonları.

Barındırılan modeller için model adını basitçe bir string (dizgi) olarak iletmeniz yeterlidir (örn. `"openai/text-embedding-3-small"`). Sınıf, API çağrılarını ve önbelleğe alma (caching) işlemlerini yönetmek için `litellm` kullanacaktır.

Özel (custom) embedding modelleri için ise şu özellikleri taşıyan çağrılabilir (callable) bir fonksiyon iletin:
- Girdi olarak bir string listesi (`list of strings`) alır.
- Embedding'leri şu iki formattan biri olarak döndürür:
  - `float32` değerlerinden oluşan 2 boyutlu (2D) bir numpy dizisi (array).
  - `float32` değerlerinden oluşan 2 boyutlu (2D) bir liste.
- Her bir satır bir embedding vektörünü temsil etmelidir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `model` | `str \| Callable` | Kullanılacak embedding modeli. Bu bir string (barındırılan embedding modelinin adını temsil eder, `litellm` tarafından desteklenen bir embedding modeli olmalıdır) veya özel bir embedding modelini temsil eden çağrılabilir (callable) bir fonksiyon olabilir. | **Gerekli (required)** |
| `batch_size` | `int` | Girdileri yığınlar (batches) halinde işlemek için varsayılan yığın boyutu. | `200` |
| `caching` | `bool` | Barındırılan bir model kullanırken embedding yanıtının önbelleğe alınıp alınmayacağı (caching). | `True` |
| `**kwargs` | `dict[str, Any]` | Embedding modeline iletilecek ek varsayılan anahtar kelime argümanları. | `{}` |

**Örnekler (Examples):**

Örnek 1: Barındırılan (hosted) bir modelin kullanımı.

```python
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
embeddings = embedder(["hello", "world"])
assert embeddings.shape == (2, 1536)
```

Örnek 2: Herhangi bir yerel (local) embedding modelinin kullanımı, örn. https://huggingface.co/models?library=sentence-transformers adresinden.

```python
# pip install sentence_transformers
import dspy
from sentence_transformers import SentenceTransformer

# Geri getirme (retrieval) işlemi için son derece verimli yerel bir model yükleyin
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
embedder = dspy.Embedder(model.encode)
embeddings = embedder(["hello", "world"], batch_size=1)
assert embeddings.shape == (2, 1024)
```

Örnek 3: Özel (custom) bir fonksiyon kullanımı.

```python
import dspy
import numpy as np

def my_embedder(texts):
    return np.random.rand(len(texts), 10)

embedder = dspy.Embedder(my_embedder)
embeddings = embedder(["hello", "world"], batch_size=1)
assert embeddings.shape == (2, 10)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/embedding.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(inputs: str | list[str], batch_size: int | None = None, caching: bool | None = None, **kwargs: dict[str, Any]) -> np.ndarray
```

Verilen girdiler için embedding'leri hesaplar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `inputs` | `str \| list[str]` | Embedding'leri hesaplanacak girdiler, tek bir string veya string'lerden oluşan bir liste olabilir. | **Gerekli (required)** |
| `batch_size` | `int` | Girdileri işlemek için kullanılacak yığın boyutu (batch size). `None` ise, başlatma sırasında ayarlanan `batch_size` varsayılan olarak kullanılır. | `None` |
| `caching` | `bool` | Barındırılan bir model kullanırken embedding yanıtının önbelleğe alınıp alınmayacağı. `None` ise, başlatma sırasındaki caching ayarı varsayılan olarak kullanılır. | `None` |
| `kwargs` | `dict[str, Any]` | Embedding modeline iletilecek ek anahtar kelime argümanları. Bunlar, başlatma sırasında sağlanan varsayılan `kwargs` argümanlarını geçersiz kılacaktır (override). | `{}` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `ndarray` | `numpy.ndarray`: Girdi tek bir string ise, embedding'i temsil eden 1 boyutlu (1D) bir numpy dizisi döndürür. Girdi bir string listesi ise, her satırda bir embedding olacak şekilde 2 boyutlu (2D) bir numpy dizisi döndürür. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/embedding.py`*

### `acall`

```python
acall(inputs, batch_size=None, caching=None, **kwargs) async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/embedding.py`*

