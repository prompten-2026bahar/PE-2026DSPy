# /translation/citations.md

## dspy.experimental.Citations

```python
dspy.experimental.Citations
```

**Kullanılan Yapılar (Bases):** `Type`

LM (Dil Modeli) yanıtından kaynak referanslarıyla birlikte çıkarılan alıntılar (citations).

Bu tip, alıntı çıkarma (citation extraction) özelliğini destekleyen dil modelleri, özellikle LiteLLM aracılığıyla Anthropic'in Citations API'si tarafından döndürülen alıntıları temsil eder. Alıntılar, alıntılanan metni ve kaynak bilgisini içerir.

**Örnekler (Examples):**

```python
import os
import dspy
from dspy.signatures import Signature
from dspy.experimental import Citations, Document

os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

class AnswerWithSources(Signature):
    '''Sağlanan belgeleri kullanarak soruları alıntılarla (citations) yanıtlar.'''
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()

# Kaynak olarak sağlanacak belgeleri oluşturun
docs = [
    Document(
        data="The Earth orbits the Sun in an elliptical path.",
        title="Basic Astronomy Facts"
    ),
    Document(
        data="Water boils at 100°C at standard atmospheric pressure.",
        title="Physics Fundamentals",
        metadata={"author": "Dr. Smith", "year": 2023}
    )
]

# Claude gibi alıntıları destekleyen bir modelle kullanın
lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
predictor = dspy.Predict(AnswerWithSources)

result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)

for citation in result.citations.citations:
    print(citation.format())
```

---

## Functions (Fonksiyonlar)

### `adapt_to_native_lm_feature`

```python
adapt_to_native_lm_feature(signature, field_name, lm, lm_kwargs) -> bool # classmethod
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `description`

```python
description() -> str # classmethod
```

İstemlerde (prompts) kullanılmak üzere alıntılar (citations) tipinin açıklaması.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `extract_custom_type_from_annotation`

```python
extract_custom_type_from_annotation(annotation) # classmethod
```

Ek açıklamadan (annotation) tüm özel tipleri (custom types) çıkarır.

Bu, açıklamanın (annotation) rastgele iç içe geçme (nesting) düzeyine sahip olabileceği durumlarda, bir alanın ek açıklamasından tüm özel tipleri çıkarmak için kullanılır. Örneğin, `Tool` yapısının `list[dict[str, Tool]]` içinde olduğunu bu şekilde tespit ederiz.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `format`

```python
format() -> list[dict[str, Any]]
```

Alıntıları bir sözlük listesi (list of dictionaries) olarak formatlar.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `from_dict_list`

```python
from_dict_list(citations_dicts: list[dict[str, Any]]) -> Citations # classmethod
```

Bir sözlük listesini bir `Citations` örneğine dönüştürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `citations_dicts` | `list[dict[str, Any]]` | Her bir sözlüğün 'cited_text' anahtarına ve 'document_index', 'start_char_index', 'end_char_index' anahtarlarına sahip olması gereken bir sözlük listesi. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Citations` | Bir `Citations` örneği (instance). |

**Örnekler (Examples):**

```python
citations_dict = [
    {
        "cited_text": "The sky is blue",
        "document_index": 0,
        "document_title": "Weather Guide",
        "start_char_index": 0,
        "end_char_index": 15,
        "supported_text": "The sky was blue yesterday."
    }
]

citations = Citations.from_dict_list(citations_dict)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `is_streamable`

```python
is_streamable() -> bool # classmethod
```

`Citations` tipinin akışa uygun (streamable) olup olmadığı bilgisini verir.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `parse_lm_response`

```python
parse_lm_response(response: str | dict[str, Any]) -> Optional[Citations] # classmethod
```

Bir LM yanıtını `Citations` tipine ayrıştırır (parse).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `response` | `str \| dict[str, Any]` | Alıntı (citation) verisi içerebilecek bir LM yanıtı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Citations]` | Alıntı verisi bulunursa bir `Citations` nesnesi, aksi takdirde `None`. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `parse_stream_chunk`

```python
parse_stream_chunk(chunk) -> Optional[Citations] # classmethod
```

Bir akış yığınını (stream chunk) `Citations` tipine ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `chunk` | - | LM'den gelen bir akış yığını. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Optional[Citations]` | Yığın (chunk) alıntı verisi içeriyorsa bir `Citations` nesnesi, aksi takdirde `None`. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

### `serialize_model`

```python
serialize_model()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/base_type.py`*

### `validate_input`

```python
validate_input(data: Any) # classmethod
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/citation.py`*

