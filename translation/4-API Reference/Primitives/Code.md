# /translation/code.md

## dspy.Code

```python
dspy.Code
```

**Kullanılan Yapılar (Bases):** `Type`

DSPy'daki kod türüdür (Code type).
Bu tür, kod üretimi (code generation) ve kod analizi için oldukça kullanışlıdır.

**Örnek 1:** Kod üretiminde çıktı (output) türü olarak `dspy.Code`:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CodeGeneration(dspy.Signature):
    '''Generate python code to answer the question.'''
    question: str = dspy.InputField(description="The question to answer")
    code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")

predict = dspy.Predict(CodeGeneration)
result = predict(question="Given an array, find if any of the two numbers sum up to 10")
print(result.code)
```

**Örnek 2:** Kod analizinde girdi (input) türü olarak `dspy.Code`:

```python
import dspy
import inspect

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CodeAnalysis(dspy.Signature):
    '''Analyze the time complexity of the function.'''
    code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
    result: str = dspy.OutputField(description="The time complexity of the function")

predict = dspy.Predict(CodeAnalysis)

def sleepsort(x):
    import time
    for i in x:
        time.sleep(i)
        print(i)

result = predict(code=inspect.getsource(sleepsort))
print(result.result)
```

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

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/code.py`*

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
format()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/code.py`*

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

`< >` ve `< >` etiketlerini atlamak (bypass) için geçersiz kılar (override).

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/code.py`*

### `validate_input`

```python
validate_input(data: Any)
```

*(Sınıf Metodu / classmethod)*

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/code.py`*
