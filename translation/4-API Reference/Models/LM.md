# /translation/lm.md

## dspy.LM

```python
dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs)
```

**Kullanılan Yapılar (Bases):** `BaseLM`

DSPy modülleriyle kullanım için sohbet (chat) veya metin tamamlama (text completion) isteklerini destekleyen bir dil modelidir (language model).

DSPy modülleri ve programlarıyla kullanılmak üzere yeni bir dil modeli örneği (instance) oluşturur.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `model` | `str` | Kullanılacak model. Bu, LiteLLM tarafından desteklenen `"llm_provider/llm_name"` formatında bir string (dizgi) olmalıdır. Örneğin, `"openai/gpt-4o"`. | **Gerekli (required)** |
| `model_type` | `Literal['chat', 'text', 'responses']` | Modelin tipi, `"chat"` veya `"text"`. | `'chat'` |
| `temperature` | `float \| None` | Yanıtlar üretilirken kullanılacak örnekleme (sampling) sıcaklığı. | `None` |
| `max_tokens` | `int \| None` | Yanıt başına üretilecek maksimum token (belirteç) sayısı. | `None` |
| `cache` | `bool` | Performansı artırmak ve maliyetleri düşürmek için model yanıtlarının yeniden kullanım amacıyla önbelleğe (cache) alınıp alınmayacağı. | `True` |
| `callbacks` | `list[BaseCallback] \| None` | Her istekten önce ve sonra çalıştırılacak geri çağırma (callback) fonksiyonlarının bir listesi. | `None` |
| `num_retries` | `int` | Ağ hatası, hız sınırlaması (rate limiting) vb. nedenlerle geçici olarak başarısız olan bir isteğin kaç kez yeniden deneneceği. İstekler üstel geri çekilme (exponential backoff) ile yeniden denenir. | `3` |
| `provider` | `Provider \| None` | Kullanılacak sağlayıcı. Belirtilmezse, sağlayıcı modelden çıkarım yoluyla (inferred) belirlenir. | `None` |
| `finetuning_model` | `str \| None` | İnce ayar (finetune) yapılacak model. Bazı sağlayıcılarda, ince ayar için kullanılabilen modeller, çıkarım (inference) için kullanılabilen modellerden farklıdır. | `None` |
| `rollout_id` | - | Aksi takdirde tamamen aynı olan istekler için önbellek girişlerini (cache entries) ayırt etmek amacıyla kullanılan isteğe bağlı tam sayı (integer). Farklı değerler DSPy'ın önbelleklerini atlar (bypass), ancak aynı girdilere ve `rollout_id` değerine sahip gelecekteki çağrıları yine de önbelleğe alır. Unutmayın ki `rollout_id`, yalnızca `temperature` sıfırdan farklı olduğunda üretimi etkiler. Bu argüman, sağlayıcıya istek gönderilmeden önce çıkarılır (stripped). | **Gerekli (required)** |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

---

## Functions (Fonksiyonlar)

### `__call__`

```python
__call__(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) -> list[dict[str, Any] | str]
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*

### `acall`

```python
acall(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) -> list[dict[str, Any] | str] async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*

### `aforward`

```python
aforward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs) async
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `copy`

```python
copy(**kwargs)
```

Dil modelinin muhtemelen güncellenmiş parametrelere sahip bir kopyasını döndürür.

Sağlanan herhangi bir anahtar kelime argümanı, kopyanın ilgili özniteliklerini (attributes) veya LM kwargs değerlerini günceller. Örneğin, `lm.copy(rollout_id=1, temperature=1.0)`, önbellek çakışmalarını (cache collisions) atlamak için sıfır olmayan sıcaklıkta farklı bir `rollout_id` kullanan istekler yapan bir LM döndürür.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `**kwargs` | - | Kopyada güncellenecek alanlar. | `{}` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*

### `dump_state`

```python
dump_state()
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `finetune`

```python
finetune(train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | None, train_kwargs: dict[str, Any] | None = None) -> TrainingJob
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `forward`

```python
forward(prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `infer_provider`

```python
infer_provider() -> Provider
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `inspect_history`

```python
inspect_history(n: int = 1, file: TextIO | None = None) -> None
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*

### `kill`

```python
kill(launch_kwargs: dict[str, Any] | None = None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `launch`

```python
launch(launch_kwargs: dict[str, Any] | None = None)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `reinforce`

```python
reinforce(train_kwargs) -> ReinforceJob
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/lm.py`*

### `update_history`

```python
update_history(entry)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/clients/base_lm.py`*
