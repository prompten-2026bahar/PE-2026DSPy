# /translation/status_message_provider.md

## dspy.streaming.StatusMessageProvider

```python
dspy.streaming.StatusMessageProvider
```

DSPy programları için özelleştirilebilir durum mesajı akışı (streaming) sağlar.

Bu sınıf, özel durum mesajı sağlayıcıları oluşturmak için bir temel (base) görevi görür. Kullanıcılar, program yürütmesinin (execution) farklı aşamaları için belirli durum mesajlarını tanımlamak üzere alt sınıflar (subclass) oluşturabilir ve metotlarını geçersiz kılabilir (override). Her metot bir dize (string) döndürmelidir.

**Örnekler (Examples):**

```python
class MyStatusMessageProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."
        
    def module_end_status_message(self, outputs):
        return f"Module finished with output: {outputs}!"

program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
```

---

## Functions (Fonksiyonlar)

### `lm_end_status_message`

```python
lm_end_status_message(outputs: Any)
```

Bir `dspy.LM` çağrıldıktan sonraki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*

### `lm_start_status_message`

```python
lm_start_status_message(instance: Any, inputs: dict[str, Any])
```

Bir `dspy.LM` çağrılmadan önceki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*

### `module_end_status_message`

```python
module_end_status_message(outputs: Any)
```

Bir `dspy.Module` veya `dspy.Predict` çağrıldıktan sonraki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*

### `module_start_status_message`

```python
module_start_status_message(instance: Any, inputs: dict[str, Any])
```

Bir `dspy.Module` veya `dspy.Predict` çağrılmadan önceki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*

### `tool_end_status_message`

```python
tool_end_status_message(outputs: Any)
```

Bir `dspy.Tool` çağrıldıktan sonraki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*

### `tool_start_status_message`

```python
tool_start_status_message(instance: Any, inputs: dict[str, Any])
```

Bir `dspy.Tool` çağrılmadan önceki durum mesajı.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/messages.py`*
