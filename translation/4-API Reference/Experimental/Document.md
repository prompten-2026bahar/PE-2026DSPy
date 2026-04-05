# /translation/document.md

## dspy.experimental.Document

```python
dspy.experimental.Document
```

**Kullanılan Yapılar (Bases):** `Type`

Dil modelleri (LM) tarafından alıntılanabilecek (cited) içerik sağlamak için kullanılan bir belge (document) türüdür.

Bu tür, alıntı destekli yanıtlar için dil modellerine aktarılabilecek belgeleri temsil eder ve özellikle Anthropic'in Citations API'si ile oldukça kullanışlıdır. Belgeler, içeriğin yanı sıra dil modelinin kaynak materyali anlamasına ve referans vermesine yardımcı olan meta verileri (metadata) de içerir.

**Öznitelikler (Attributes):**

| İsim (Name) | Tip (Type) | Açıklama (Description) |
| :--- | :--- | :--- |
| `data` | `str` | Belgenin metin içeriği. |
| `title` | `str \| None` | Belge için isteğe bağlı başlık (alıntılarda kullanılır). |
| `media_type` | `Literal['text/plain', 'application/pdf']` | Belge içeriğinin MIME türü (varsayılan olarak "text/plain"). |
| `context` | `str \| None` | Belge hakkında isteğe bağlı bağlam (context) bilgisi. |

**Örnekler (Examples):**

```python
import dspy
from dspy.signatures import Signature
from dspy.experimental import Document, Citations

class AnswerWithSources(Signature):
    '''Sağlanan belgeleri kullanarak soruları alıntılarla (citations) yanıtlar.'''
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()

# Belgeleri oluşturun
docs = [
    Document(
        data="The Earth orbits the Sun in an elliptical path.",
        title="Basic Astronomy Facts"
    ),
    Document(
        data="Water boils at 100°C at standard atmospheric pressure.",
        title="Physics Fundamentals",
    )
]

# Alıntı destekleyen bir modelle kullanın
lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
predictor = dspy.Predict(AnswerWithSources)
result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)

print(result.citations)
```

---

## Functions (Fonksiyonlar)

### `adapt_to_native_lm_feature`

```python
adapt_to_native_lm_feature(signature: type[Signature], field_name: str, lm: LM, lm_kwargs: dict[str, Any]) -> type[Signature] # classmethod
```

Özel türü (custom type) mümkünse yerel (native) LM özelliğine uyarlar.
LM ve yapılandırma ilgili yerel LM özelliğini (örn. yerel araç çağırma, yerel akıl yürütme vb.) desteklediğinde, yerel LM özelliğini etkinleştirmek için imzayı (signature) ve `lm_kwargs` argümanlarını güncelleriz.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | LM çağrısı için DSPy imzası. | **Gerekli (required)** |
| `field_name` | `str` | İmzadaki, yerel LM özelliğine uyarlanacak alanın adı. | **Gerekli (required)** |
| `lm` | `LM` | LM örneği (instance). | **Gerekli (required)** |
| `lm_kwargs` | `dict[str, Any]` | LM çağrısı için anahtar kelime argümanları (uyarlama gerekirse yerinde güncellemeler (in-place updates) yapılabilir). | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `type[Signature]` | Uyarlanmış imza. Eğer özel tür LM tarafından yerel olarak desteklenmiyorsa, orijinal imzayı döndürür. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `description`

```python
description() -> str # classmethod
```

İstemlerde (prompts) kullanılmak üzere belge (document) türünün açıklaması.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/document.py`*

### `extract_custom_type_from_annotation`

```python
extract_custom_type_from_annotation(annotation) # classmethod
```

Ek açıklamadan (annotation) tüm özel türleri (custom types) çıkarır.
Bu, açıklamanın (annotation) rastgele iç içe geçme (nesting) düzeyine sahip olabileceği durumlarda, bir alanın ek açıklamasından tüm özel tipleri çıkarmak için kullanılır. Örneğin, `Tool` yapısının `list[dict[str, Tool]]` içinde olduğunu bu şekilde tespit ederiz.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `format`

```python
format() -> list[dict[str, Any]]
```

Belgeyi LM tüketimi (kullanımı) için formatlar.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | Alıntı destekli dil modelleri tarafından beklenen formattaki belge bloğunu içeren bir liste. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/document.py`*

### `is_streamable`

```python
is_streamable() -> bool # classmethod
```

Özel türün (custom type) akışa uygun (streamable) olup olmadığını belirtir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `parse_lm_response`

```python
parse_lm_response(response: str | dict[str, Any]) -> Optional[Type] # classmethod
```

Bir LM yanıtını özel türe ayrıştırır (parse).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `response` | `str \| dict[str, Any]` | Bir LM yanıtı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Type]` | Özel türde (custom type) bir nesne. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `parse_stream_chunk`

```python
parse_stream_chunk(chunk: ModelResponseStream) -> Optional[Type] # classmethod
```

Bir akış yığınını (stream chunk) özel türe ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `chunk` | `ModelResponseStream` | Bir akış yığını. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Type]` | Özel türde (custom type) bir nesne veya eğer yığın bu özel tür için değilse `None`. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `serialize_model`

```python
serialize_model()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `validate_input`

```python
validate_input(data: Any) # classmethod
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/document.py`*

