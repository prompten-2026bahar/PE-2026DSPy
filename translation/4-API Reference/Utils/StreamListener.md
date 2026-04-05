# /translation/stream_listener.md

## dspy.streaming.StreamListener

```python
dspy.streaming.StreamListener(signature_field_name: str, predict: Any = None, predict_name: str | None = None, allow_reuse: bool = False)
```

Bir tahmincinin (predictor) belirli bir çıktı alanının (output field) akışını (streaming) yakalamak için akışı dinleyen sınıf.



**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `signature_field_name` | `str` | Dinlenecek alanın (field) adı. | **Gerekli (required)** |
| `predict` | `Any` | Dinlenecek tahminci (predictor). Eğer `None` ise, `streamify()` çağrıldığında imzasında `signature_field_name` bulunan tahminciyi otomatik olarak arayacaktır. | `None` |
| `predict_name` | `str \| None` | Dinlenecek tahmincinin adı. Eğer `None` ise, `streamify()` çağrıldığında imzasında `signature_field_name` bulunan tahminciyi otomatik olarak arayacaktır. | `None` |
| `allow_reuse` | `bool` | `True` ise, akış dinleyicisi (stream listener) birden fazla akış için yeniden kullanılabilir. Aynı akış parçasının (stream chunk) birden fazla dinleyiciye gönderilmesinin performansa zarar verebileceğini lütfen unutmayın. | `False` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/streaming_listener.py`*

---

## Functions (Fonksiyonlar)

### `finalize`

```python
finalize() -> StreamResponse | None
```

Akışı sonlandırır ve arabellekte (buffer) kalan token'ları boşaltır (flush).
Bu işlem akış sona erdiğinde çağrılmalıdır. Arabellekten hiçbir token'ın kaybolmamasını sağlar ve son parçayı (chunk) uygun şekilde işaretler.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `StreamResponse \| None` | Kalan arabelleğe alınmış token'lar ve `is_last_chunk=True` ile bir `StreamResponse`, veya arabellekte token yoksa ya da akış başlamadıysa `None`. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/streaming_listener.py`*

### `flush`

```python
flush() -> str
```

Alan sonu (field end) kuyruğundaki tüm token'ları boşaltır.
Bu metot, akış sona erdiğinde son birkaç token'ı boşaltmak için çağrılır. Bu token'lar arabellektedir çünkü `ChatAdapter` için `"[[ ## ... ## ]]"` gibi `end_identifier` (bitiş tanımlayıcısı) token'larını üretmemek (yield etmemek) amacıyla, akış dinleyicisi tarafından alınan token'ları doğrudan üretmeyiz.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/streaming_listener.py`*

### `receive`

```python
receive(chunk: ModelResponseStream)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/streaming_listener.py`*
