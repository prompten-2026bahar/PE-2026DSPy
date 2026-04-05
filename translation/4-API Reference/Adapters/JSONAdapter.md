# /translation/json_adapter.md

## dspy.JSONAdapter

```python
dspy.JSONAdapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True)
```

**Kullanılan Yapılar (Bases):** `ChatAdapter`

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `acall`

```python
acall(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]] async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `format`

```python
format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
```

LM çağrısı için girdi mesajlarını formatlar.

Bu metot, DSPy yapılandırılmış girdisini, az örnekli (few-shot) örnekler ve konuşma geçmişi (conversation history) ile birlikte LM tarafından beklenen çok turlu (multiturn) mesajlara dönüştürür. Özel adaptörler (custom adapters) için, girdi mesajlarının formatlanmasını özelleştirmek amacıyla bu metot geçersiz kılınabilir (overridden).

Genel olarak mesajların aşağıdaki yapıya (structure) sahip olmasını öneririz:

```json
[
    {"role": "system", "content": "system_message"},
    # Az örnekli (few-shot) örneklerin başlangıcı
    {"role": "user", "content": "few_shot_example_1_input"},
    {"role": "assistant", "content": "few_shot_example_1_output"},
    {"role": "user", "content": "few_shot_example_2_input"},
    {"role": "assistant", "content": "few_shot_example_2_output"},
    ...
    # Az örnekli örneklerin sonu
    # Konuşma geçmişinin başlangıcı
    {"role": "user", "content": "conversation_history_1_input"},
    {"role": "assistant", "content": "conversation_history_1_output"},
    {"role": "user", "content": "conversation_history_2_input"},
    {"role": "assistant", "content": "conversation_history_2_output"},
    ...
    # Konuşma geçmişinin sonu
    {"role": "user", "content": "current_input"}
]
```
Ve sistem mesajı (system message) alan açıklamasını, alan yapısını ve görev açıklamasını içermelidir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Girdi mesajlarının formatlanacağı DSPy imzası. | **Gerekli (required)** |
| `demos` | `list[dict[str, Any]]` | Az örnekli (few-shot) örneklerin listesi. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | DSPy modülüne gönderilen girdi argümanları. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | LM tarafından beklenen çok turlu (multiturn) mesajların listesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_assistant_message_content`

```python
format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message=None) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `format_conversation_history`

```python
format_conversation_history(signature: type[Signature], history_field_name: str, inputs: dict[str, Any]) -> list[dict[str, Any]]
```

Konuşma geçmişini formatlar.

Bu metot, konuşma geçmişini ve mevcut girdiyi çok turlu (multiturn) mesajlar olarak formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Konuşma geçmişinin formatlanacağı DSPy imzası. | **Gerekli (required)** |
| `history_field_name` | `str` | İmzadaki geçmiş alanının (history field) adı. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | DSPy modülüne gönderilen girdi argümanları. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | Çok turlu mesajların listesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_demos`

```python
format_demos(signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]
```

Az örnekli (few-shot) örnekleri formatlar.

Bu metot, az örnekli örnekleri çok turlu (multiturn) mesajlar olarak formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Örneklerin formatlanacağı DSPy imzası. | **Gerekli (required)** |
| `demos` | `list[dict[str, Any]]` | Az örnekli örneklerin listesi, her bir eleman imzanın girdi ve çıktı alanlarının anahtarlarını içeren bir sözlüktür. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | Çok turlu mesajların listesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_field_description`

```python
format_field_description(signature: type[Signature]) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/chat_adapter.py`*

### `format_field_structure`

```python
format_field_structure(signature: type[Signature]) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `format_field_with_value`

```python
format_field_with_value(fields_with_values: dict[FieldInfoWithName, Any], role: str = 'user') -> str
```

Belirtilen alanların (fields) değerlerini, alanın DSPy tipine (input veya output), ek açıklamasına (annotation) (örn. str, int vb.) ve değerin kendi tipine göre formatlar. Formatlanmış değerleri, birden fazla alan varsa çok satırlı bir string olan tek bir string (dizgi) halinde birleştirir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `fields_with_values` | `dict[FieldInfoWithName, Any]` | Bir alan hakkındaki bilgileri karşılık gelen değerine eşleyen bir sözlük. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `str` | Alanların birleştirilmiş formatlanmış değerleri, bir string olarak temsil edilir. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `format_finetune_data`

```python
format_finetune_data(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, list[Any]]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `format_system_message`

```python
format_system_message(signature: type[Signature]) -> str
```

LM çağrısı için sistem mesajını formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Sistem mesajının formatlanacağı DSPy imzası. | **Gerekli (required)** |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_task_description`

```python
format_task_description(signature: type[Signature]) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/chat_adapter.py`*

### `format_user_message_content`

```python
format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/chat_adapter.py`*

### `parse`

```python
parse(signature: type[Signature], completion: str) -> dict[str, Any]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*

### `user_message_output_requirements`

```python
user_message_output_requirements(signature: type[Signature]) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/json_adapter.py`*
