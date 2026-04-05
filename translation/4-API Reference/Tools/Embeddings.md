# /translation/embeddings.md

## dspy.retrievers.Embeddings

```python
dspy.Embeddings(corpus: list[str], embedder, k: int = 5, callbacks: list[Any] | None = None, cache: bool = False, brute_force_threshold: int = 20000, normalize: bool = True)
```

DSPy Gömme (Embeddings) getiricisi (retriever).

Bu sınıf, gömme (embedding) tabanlı benzerlik araması kullanarak bir derlemden (corpus) en benzer ilk `k` (top-k) pasajı getirir. Büyük derlemler için, hızlı yaklaşık aday getirme (approximate candidate retrieval) ve ardından kesin yeniden sıralama (exact re-ranking) amacıyla bir FAISS indeksi oluşturulur. Küçük derlemler için kaba kuvvet (brute-force) araması kullanılır.



*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(query: str)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*

### `forward`

```python
forward(query: str)
```

Sorguya en benzer ilk `k` pasajı arar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `query` | `str` | Arama sorgusu metni (dizesi). | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| - | `dspy.Prediction`: Pasajları ve bunların derlem indekslerini içeren bir tahmin. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*

### `from_saved`

```python
from_saved(path: str, embedder)
```

*(Sınıf Metodu / classmethod)*

Kaydedilmiş bir indeksten bir Embeddings örneği oluşturur.
Gömme işlemlerini gereksiz yere hesaplamadan yeni bir örnek oluşturduğu için, kaydedilmiş gömmeleri (embeddings) yüklemenin önerilen yolu budur.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | Gömmelerin kaydedildiği dizin yolu. | **Gerekli (required)** |
| `embedder` | - | Yeni sorgular için kullanılacak gömücü (embedder) fonksiyon. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| - | Diske kaydedilmiş Embeddings örneği. |

**Örnekler (Examples):**

```python
# Gömmeleri kaydet
embeddings = Embeddings(corpus, embedder)
embeddings.save("./saved_embeddings")

# Gömmeleri daha sonra yükle
loaded_embeddings = Embeddings.from_saved("./saved_embeddings", embedder)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*

### `load`

```python
load(path: str, embedder)
```

Gömmeler indeksini diskten mevcut örneğin (instance) içine yükler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | Gömmelerin kaydedildiği dizin yolu. | **Gerekli (required)** |
| `embedder` | - | Yeni sorgular için kullanılacak gömücü fonksiyon. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| İsim (Name) | Tip (Type) | Açıklama (Description) |
| :--- | :--- | :--- |
| `self` | - | Metot zincirleme (method chaining) için `self` döndürür. |

**Hatalar (Raises):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `FileNotFoundError` | Kayıt dizini veya gerekli dosyalar mevcut değilse fırlatılır. |
| `ValueError` | Kaydedilen yapılandırma geçersiz veya uyumsuzsa fırlatılır. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*

### `save`

```python
save(path: str)
```

Gömmeler indeksini diske kaydeder.
Bu işlem; derlemi, gömmeleri, FAISS indeksini (eğer mevcutsa) ve gömmeleri yeniden hesaplamadan hızlı yüklemeye olanak tanıyan yapılandırmayı kaydeder.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `path` | `str` | Gömmelerin kaydedileceği dizin yolu. | **Gerekli (required)** |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/retrievers/embeddings.py`*
