# /translation/image.md

## dspy.Image

```python
dspy.Image(url: Any = None, *, download: bool = False, verify: bool = True, **data)
```

**Kullanılan Yapılar (Bases):** `Type`

Bir Görüntü (Image) oluşturur.

### Parametreler (Parameters)

* **`url`**: Görüntü kaynağı. Desteklenen değerler şunları içerir:
    * `str`: HTTP(S)/GS URL'si veya yerel dosya yolu
    * `bytes`: Ham (raw) görüntü baytları
    * `PIL.Image.Image`: Bir PIL görüntü örneği (instance)
    * `dict`: Tek bir `{"url": value}` girdisine sahip sözlük (eski form / legacy form)
    * Zaten kodlanmış veri URI'si (data URI)
* **`download`**: MIME türlerini anlamak (infer) için uzak URL'lerin indirilip indirilmeyeceğini belirler.
* **`verify`**: URL'lerden görüntü indirirken SSL sertifikalarının doğrulanıp doğrulanmayacağını belirler. Kendinden imzalı (self-signed) sertifikalar için `False` olarak ayarlayın. Varsayılan `True`'dur.
* Diğer tüm ek anahtar kelime argümanları `pydantic.BaseModel` sınıfına aktarılır.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/image.py`*

---

## Functions (Fonksiyonlar)

### `adapt_to_native_lm_feature`

```python
adapt_to_native_lm_feature(signature: type[Signature], field_name: str, lm: LM, lm_kwargs: dict[str, Any]) -> type[Signature]
```

*(Sınıf Metodu / classmethod)*

Mümkünse özel türü (custom type), yerel (native) LM özelliğine uyarlar.
LM ve yapılandırma ilgili yerel LM özelliğini (örn. yerel araç çağırma, yerel akıl yürütme vb.) desteklediğinde, yerel LM özelliğini etkinleştirmek için imzayı (signature) ve `lm_kwargs`'ı uyarlarız.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | LM çağrısı için DSPy imzası. | **Gerekli (required)** |
| `field_name` | `str` | İmzada yerel LM özelliğine uyarlanacak alanın (field) adı. | **Gerekli (required)** |
| `lm` | `LM` | LM örneği (instance). | **Gerekli (required)** |
| `lm_kwargs` | `dict[str, Any]` | LM çağrısı için anahtar kelime argümanları; uyarlama gerekiyorsa yerinde (in-place) güncellemeler yapılabilir. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Uyarlanmış imza. Eğer özel tür LM tarafından yerel olarak desteklenmiyorsa, orijinal `type[Signature]` imzasını döndürür. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `description`

```python
description() -> str
```

*(Sınıf Metodu / classmethod)*

Özel türün (custom type) açıklaması.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `extract_custom_type_from_annotation`

```python
extract_custom_type_from_annotation(annotation)
```

*(Sınıf Metodu / classmethod)*

Ek açıklamadan (annotation) tüm özel türleri çıkarır.
Bu, bir alanın (field) ek açıklamasındaki tüm özel türleri çıkarmak için kullanılır, ancak ek açıklama isteğe bağlı düzeyde iç içe geçme (nesting) içerebilir. Örneğin, `Tool` yapısının `list[dict[str, Tool]]` içinde olduğunu tespit edebiliriz.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `format`

```python
format() -> list[dict[str, Any]] | str
```

*(Önbelleğe alınmış / cached)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/image.py`*

### `from_PIL`

```python
from_PIL(pil_image)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/image.py`*

### `from_file`

```python
from_file(file_path: str)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/image.py`*

### `from_url`

```python
from_url(url: str, download: bool = False)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/image.py`*

### `is_streamable`

```python
is_streamable() -> bool
```

*(Sınıf Metodu / classmethod)*

Özel türün akışa uygun (streamable) olup olmadığı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `parse_lm_response`

```python
parse_lm_response(response: str | dict[str, Any]) -> Optional[Type]
```

*(Sınıf Metodu / classmethod)*

Bir LM yanıtını özel türe ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `response` | `str \| dict[str, Any]` | Bir LM yanıtı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Type]` | Bir özel tür (custom type) nesnesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `parse_stream_chunk`

```python
parse_stream_chunk(chunk: ModelResponseStream) -> Optional[Type]
```

*(Sınıf Metodu / classmethod)*

Bir akış parçasını (stream chunk) özel türe ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `chunk` | `ModelResponseStream` | Bir akış parçası. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Type]` | Bir özel tür nesnesi veya parça bu özel türe ait değilse `None`. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `serialize_model`

```python
serialize_model()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*
