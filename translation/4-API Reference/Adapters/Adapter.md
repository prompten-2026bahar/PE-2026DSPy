# Adapters (Adaptörler)

## dspy.Adapter

```python
dspy.Adapter(callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False, native_response_types: list[type[Type]] | None = None)
```

Temel Adaptör (Base Adapter) sınıfı.

Adaptör, DSPy modülü/imzası (Signature) ile Dil Modelleri (Language Models - LMs) arasındaki arayüz katmanı olarak hizmet eder. DSPy girdilerinden LM çağrılarına ve tekrar yapılandırılmış çıktılara (structured outputs) kadar olan tüm dönüşüm boru hattını yönetir.

**Temel Sorumluluklar (Key responsibilities):**
* Kullanıcı girdilerini ve imzaları (Signatures), LM'yi yanıtı belirli bir formatta vermesi için de yönlendiren, düzgün formatlanmış LM istemlerine (Prompts) dönüştürmek.
* LM çıktılarını, imzanın çıktı alanlarıyla (output fields) eşleşen sözlüklere (dictionaries) ayrıştırmak (Parsing).
* Konfigürasyona bağlı olarak yerel (native) LM özelliklerini (fonksiyon çağırma, alıntılar/citations vb.) etkinleştirmek/devre dışı bırakmak.
* Konuşma geçmişini (conversation history), az örnekli (few-shot) örnekleri ve özel tip işlemlerini yönetmek.

Adaptör deseni (adapter pattern), kullanıcılar için tutarlı bir programlama modelini korurken DSPy'ın farklı LM arayüzleriyle çalışmasına olanak tanır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `callbacks` | `list[BaseCallback] \| None` | `format()` ve `parse()` metodları sırasında yürütülecek geri çağırma (callback) fonksiyonlarının listesi. Kayıt (logging), izleme (monitoring) veya özel işlemler için kullanılabilir. Varsayılan `None` (boş liste). | `None` |
| `use_native_function_calling` | `bool` | LM desteklediğinde yerel fonksiyon çağırma (native function calling) yeteneklerinin etkinleştirilip etkinleştirilmeyeceği. `True` ise, girdi alanları `dspy.Tool` veya `list[dspy.Tool]` tiplerini içerdiğinde adaptör otomatik olarak fonksiyon çağırmayı yapılandırır. Varsayılan `False`. | `False` |
| `native_response_types` | `list[type[Type]] \| None` | Adaptör ayrıştırması (parsing) yerine yerel LM özellikleri tarafından işlenmesi gereken çıktı alanı tiplerinin (output field types) listesi. Örneğin, `dspy.Citations` doğrudan alıntı API'leri (örn. Anthropic'in alıntı özelliği) tarafından doldurulabilir. Varsayılan `[Citations]`. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

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

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

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
| `signature` | `type[Signature]` | Girdi mesajlarının formatlanacağı DSPy imzası (Signature). | **Gerekli (required)** |
| `demos` | `list[dict[str, Any]]` | Az örnekli (few-shot) örneklerin listesi. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | DSPy modülüne gönderilen girdi argümanları. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `list[dict[str, Any]]` | LM tarafından beklenen çok turlu (multiturn) mesajların listesi. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_assistant_message_content`

```python
format_assistant_message_content(signature: type[Signature], outputs: dict[str, Any], missing_field_message: str | None = None) -> str
```

Asistan mesaj içeriğini formatlar.

Bu metot, az örnekli (few-shot) örnekleri ve konuşma geçmişini formatlamada kullanılabilecek asistan mesaj içeriğini formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Asistan mesaj içeriğinin formatlanacağı DSPy imzası (Signature). | **Gerekli (required)** |
| `outputs` | `dict[str, Any]` | Formatlanacak çıktı alanları. | **Gerekli (required)** |
| `missing_field_message` | `str \| None` | Bir alan eksik olduğunda kullanılacak mesaj. | `None` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `str` | Asistan mesaj içeriğini barındıran string (dizgi). |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

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
| `list[dict[str, Any]]` | Çok turlu (multiturn) mesajların listesi. |

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
| `list[dict[str, Any]]` | Çok turlu (multiturn) mesajların listesi. |

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

LM çağrısı için sistem mesajını (system message) formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Sistem mesajının formatlanacağı DSPy imzası. | **Gerekli (required)** |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_task_description`

```python
format_task_description(signature: type[Signature]) -> str
```

Sistem mesajı için görev açıklamasını (task description) formatlar.

Bu metot, sistem mesajı için görev açıklamasını formatlar. Çoğu durumda bu, yalnızca `signature.instructions` üzerinde ince bir sarmalayıcıdır (wrapper).

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | DSpy modülünün DSPy imzası. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `str` | Görevi açıklayan string. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `format_user_message_content`

```python
format_user_message_content(signature: type[Signature], inputs: dict[str, Any], prefix: str = '', suffix: str = '', main_request: bool = False) -> str
```

Kullanıcı mesaj içeriğini (user message content) formatlar.

Bu metot, az örnekli örnekleri, konuşma geçmişini ve mevcut girdiyi formatlamada kullanılabilecek kullanıcı mesaj içeriğini formatlar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | Kullanıcı mesaj içeriğinin formatlanacağı DSPy imzası. | **Gerekli (required)** |
| `inputs` | `dict[str, Any]` | DSPy modülüne gönderilen girdi argümanları. | **Gerekli (required)** |
| `prefix` | `str` | Kullanıcı mesaj içeriğine eklenecek ön ek (prefix). | `''` |
| `suffix` | `str` | Kullanıcı mesaj içeriğine eklenecek son ek (suffix). | `''` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `str` | Kullanıcı mesaj içeriğini barındıran string. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*

### `parse`

```python
parse(signature: type[Signature], completion: str) -> dict[str, Any]
```

LM çıktısını, çıktı alanlarından (output fields) oluşan bir sözlüğe ayrıştırır (parse).

Bu metot, LM çıktısını çıktı alanlarından oluşan bir sözlüğe ayrıştırır.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature` | `type[Signature]` | LM çıktısının ayrıştırılacağı DSPy imzası. | **Gerekli (required)** |
| `completion` | `str` | Ayrıştırılacak LM çıktısı. | **Gerekli (required)** |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `dict[str, Any]` | Çıktı alanlarını barındıran sözlük. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/adapters/base.py`*
