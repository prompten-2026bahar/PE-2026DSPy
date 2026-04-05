# /translation/tool.md

## dspy.Tool

```python
dspy.Tool(func: Callable, name: str | None = None, desc: str | None = None, args: dict[str, Any] | None = None, arg_types: dict[str, Any] | None = None, arg_desc: dict[str, str] | None = None)
```

**Kullanılan Yapılar (Bases):** `Type`

Araç (Tool) sınıfı.
Bu sınıf, Yüksek Dil Modellerinde (LLM'lerde) araç çağırma (tool calling / function calling) için araçların oluşturulmasını basitleştirmek amacıyla kullanılır. Şu an için sadece fonksiyonları desteklemektedir.



`Tool` sınıfını başlatır.
Kullanıcılar `name`, `desc`, `args` ve `arg_types` parametrelerini belirlemeyi seçebilir veya `dspy.Tool`'un bu değerleri fonksiyondan otomatik olarak çıkarmasına (infer) izin verebilir. Kullanıcı tarafından açıkça belirtilen değerler için otomatik çıkarma işlemi uygulanmayacaktır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `func` | `Callable` | Araç tarafından sarmalanan (wrapped) asıl fonksiyon. | **Gerekli (required)** |
| `name` | `Optional[str]` | Aracın adı. | `None` |
| `desc` | `Optional[str]` | Aracın açıklaması. | `None` |
| `args` | `Optional[dict[str, Any]]` | Aracın argümanları ve şeması (schema); argüman adından argümanın JSON şemasına eşlenen bir sözlük olarak temsil edilir. | `None` |
| `arg_types` | `Optional[dict[str, Any]]` | Aracın argüman türleri; argüman adından argümanın türüne eşlenen bir sözlük olarak temsil edilir. | `None` |
| `arg_desc` | `Optional[dict[str, str]]` | Her bir argüman için açıklamalar; argüman adından açıklama metnine eşlenen bir sözlük olarak temsil edilir. | `None` |

**Örnekler (Examples):**

```python
def foo(x: int, y: str = "hello"):
    return str(x) + y

tool = Tool(foo)
print(tool.args)
# Beklenen çıktı: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(**kwargs)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

### `acall`

```python
async acall(**kwargs)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

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

Özel türün açıklaması.

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
format()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

### `format_as_litellm_function_call`

```python
format_as_litellm_function_call()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

### `from_langchain`

```python
from_langchain(tool: BaseTool) -> Tool
```

*(Sınıf Metodu / classmethod)*

Bir LangChain aracından bir DSPy aracı oluşturur.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `tool` | `BaseTool` | Dönüştürülecek LangChain aracı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Tool` | Bir `Tool` (Araç) nesnesi. |

**Örnekler (Examples):**

```python
import asyncio
import dspy
from langchain.tools import tool as lc_tool

@lc_tool
def add(x: int, y: int):
    "Add two numbers together."
    return x + y

dspy_tool = dspy.Tool.from_langchain(add)

async def run_tool():
    return await dspy_tool.acall(x=1, y=2)

print(asyncio.run(run_tool()))
# Çıktı: 3
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

### `from_mcp_tool`

```python
from_mcp_tool(session: mcp.ClientSession, tool: mcp.types.Tool) -> Tool
```

*(Sınıf Metodu / classmethod)*

Bir MCP aracından ve bir `ClientSession`'dan bir DSPy aracı oluşturur.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `session` | `ClientSession` | Kullanılacak MCP oturumu. | **Gerekli (required)** |
| `tool` | `Tool` | Dönüştürülecek MCP aracı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Tool` | Bir `Tool` (Araç) nesnesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/types/tool.py`*

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
