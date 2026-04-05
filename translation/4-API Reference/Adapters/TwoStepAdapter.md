# /translation/twostep_adapter.md

## dspy.TwoStepAdapter

```python
dspy.TwoStepAdapter(extraction_model: LM, **kwargs)
```

**Kullanılan Yapılar (Bases):** `Adapter`

İki aşamalı (two-stage) bir adaptördür:
* Ana LM (Language Model) için daha basit, daha doğal bir istem (prompt) kullanır.
* Ana LM'nin yanıtından yapılandırılmış veriyi (structured data) çıkarmak (extract) için sohbet adaptörüne (chat adapter) sahip daha küçük bir LM kullanır.

Bu adaptör, temel `Adapter` sınıfında tanımlanan ortak `call` mantığını kullanır. Akıl yürütme modellerinin (reasoning models) yapılandırılmış çıktılarla (structured outputs) mücadele ettiği bilindiğinden, bu sınıf özellikle akıl yürütme modelleriyle ana LM olarak etkileşim kurarken son derece kullanışlıdır.

**Örnekler (Examples):**

```python
import dspy
lm = dspy.LM(model="openai/o3-mini", max_tokens=16000, temperature = 1.0)
adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
dspy.configure(lm=lm, adapter=adapter)
program = dspy.ChainOfThought("question->answer")
result = program("What is the capital of France?")
print(result)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
```

Adaptör boru hattını (pipeline) çalıştırır: Girdileri formatlar, LM'yi çağırır ve çıktıları ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `lm` | `LM` | Üretim (generation) için kullanılacak Dil Modeli örneği. `dspy.BaseLM` örneği (instance) olmalıdır. | **Gerekli (required)** |
| `lm_kwargs` | `dict[str, Any]` | LM çağrısına aktarılacak ek anahtar kelime argümanları (örn. temperature, max_tokens). Bunlar doğrudan LM'ye iletilir. | **Gerekli (required)** |
| `signature` | `type[Signature]` | Bu LM çağrısı ile ilişkili DSPy imzası (Signature). | **Gerekli (required)** |
| `demos` | `list[dict[str, Any]]` | İsteme (Prompt) dahil edilecek az örnekli (few-shot) örneklerin listesi. Her bir sözlük, imzanın girdi ve çıktı alan isimleriyle eşleşen anahtarlar içermelidir. Örnekler kullanıcı/asistan mesaj çiftleri olarak formatlanır. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | Bu çağrı için mevcut girdi değerleri. Anahtarlar, imzanın girdi alan isimleriyle eşleşmelidir. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | Ayrıştırılmış (parsed) LM yanıtlarını temsil eden sözlük listesi. Her bir sözlük, imzanın çıktı alan isimleriyle eşleşen anahtarlar içerir. Çoklu üretimler (n > 1) için birden fazla sözlük döndürür. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `acall`

```python
acall(lm: LM, lm_kwargs: dict[str, Any], signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]] async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

### `format`

```python
format(signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]
```

Ana LM ile ilk aşama (first stage) için bir istem (prompt) formatlar. Ana LM için belirli bir yapı gerekmez, `format_field_description` veya `format_field_structure` yerine `format` metodunu özelleştiririz.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Orijinal görevin imzası. | **Gerekli (required)** |
| `demos` | `list[dict[str, Any]]` | Demo (az örnekli) örneklerin listesi. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | Mevcut girdi. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | Ana LM'ye iletilecek mesajların listesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

### `format_assistant_message_content`

```python
format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

### `format_conversation_history`

```python
format_conversation_history(signature: type[Signature], history_field_name: str, inputs: dict[str, Any]) -> list[dict[str, Any]]
```

Konuşma geçmişini formatlar.

Bu metot, konuşma geçmişini (conversation history) ve mevcut girdiyi çok turlu (multiturn) mesajlar olarak formatlar.

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

Sistem mesajı (system message) için alan açıklamasını (field description) formatlar.

Bu metot, sistem mesajı için alan açıklamasını formatlar. Girdi alanları ve çıktı alanları için alan açıklamasını içeren bir string (dizgi) döndürmelidir.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Alan açıklamasının formatlanacağı DSPy imzası. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `str` | Girdi ve çıktı alanları için alan açıklamasını içeren string. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_field_structure`

```python
format_field_structure(signature: type[Signature]) -> str
```

Sistem mesajı için alan yapısını (field structure) formatlar.

Bu metot, sistem mesajı için alan yapısını formatlar. Girdi alanlarının LM'ye hangi formatta sağlanacağını ve yanıt içindeki çıktı alanlarının hangi formatta olacağını dikte eden bir string döndürmelidir. Örnek için ChatAdapter ve JsonAdapter'a başvurunuz.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Alan yapısının formatlanacağı DSPy imzası. | **Gerekli (required)** |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

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
format_task_description(signature: Signature) -> str
```

İmzaya (signature) dayalı olarak görevin bir açıklamasını oluşturur.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

### `format_user_message_content`

```python
format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '') -> str
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

### `parse`

```python
parse(signature: Signature, completion: str) -> dict[str, Any]
```

Ana LM'nin ham tamamlama metninden (raw completion text) yapılandırılmış veriyi çıkarmak için sohbet adaptörlü (chat adapter) daha küçük bir LM (`extraction_model`) kullanır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `Signature` | Orijinal görevin imzası. | **Gerekli (required)** |
| `completion` | `str` | Ana LM'den gelen tamamlama (completion) metni. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `dict[str, Any]` | Çıkarılan yapılandırılmış veriyi (extracted structured data) içeren sözlük. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/two_step_adapter.py`*

