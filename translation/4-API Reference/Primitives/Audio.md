# /translation/audio.md

## dspy.Audio

```python
dspy.Audio
```

**Kullanılan Yapılar (Bases):** `Type`

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
format() -> list[dict[str, Any]]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/audio.py`*

### `from_array`

```python
from_array(array: Any, sampling_rate: int, format: str = 'wav') -> Audio
```

*(Sınıf Metodu / classmethod)*

Numpy benzeri diziyi işler ve onu base64 olarak kodlar. Kodlama için örnekleme hızını (sampling rate) ve ses biçimini (audio format) kullanır.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/audio.py`*

### `from_file`

```python
from_file(file_path: str) -> Audio
```

*(Sınıf Metodu / classmethod)*

Yerel ses dosyasını okur ve onu base64 olarak kodlar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/audio.py`*

### `from_url`

```python
from_url(url: str) -> Audio
```

*(Sınıf Metodu / classmethod)*

URL'den bir ses dosyasını indirir ve onu base64 olarak kodlar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/audio.py`*

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

### `validate_input`

```python
validate_input(values: Any) -> Any
```

*(Sınıf Metodu / classmethod)*

Ses (Audio) için girdiyi doğrular, sözlükte 'data' ve 'audio_format' anahtarlarının bulunmasını bekler.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/audio.py`*
