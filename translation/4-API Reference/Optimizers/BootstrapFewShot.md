# /translation/bootstrap_few_shot.md

## dspy.BootstrapFewShot

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings: dict | None = None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=None)
```

**Kullanılan Yapılar (Bases):** `Teleprompter`

Bir tahmincinin (predictor) istemine (prompt) girecek demoları/örnekleri (demos/examples) oluşturan bir Teleprompter sınıfıdır. Bu demolar, eğitim setindeki etiketli örneklerin (labeled examples) ve önyüklemeli (bootstrapped) demoların bir kombinasyonundan gelir.



Her önyükleme (bootstrap) turu, önbellekleri (caches) atlamak ve çeşitli izler (traces) toplamak için LM'yi `temperature=1.0` ayarında yeni bir `rollout_id` ile kopyalar.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `metric` | `Callable` | Beklenen değer ile tahmin edilen değeri karşılaştıran ve bu karşılaştırmanın sonucunu veren bir fonksiyon. | `None` |
| `metric_threshold` | `float` | Eğer metrik sayısal bir değer veriyorsa, bir önyükleme (bootstrap) örneğinin kabul edilip edilmeyeceğine karar verirken bunu bu eşik değerine (threshold) göre kontrol edin. Varsayılan None'dır. | `None` |
| `teacher_settings` | `dict` | Öğretmen (`teacher`) modeli için ayarlar. Varsayılan None'dır. | `None` |
| `max_bootstrapped_demos` | `int` | Dahil edilecek maksimum önyüklemeli (bootstrapped) gösterim/demo sayısı. Varsayılan 4'tür. | `4` |
| `max_labeled_demos` | `int` | Dahil edilecek maksimum etiketli gösterim/demo sayısı. Varsayılan 16'dır. | `16` |
| `max_rounds` | `int` | Eğitim örneği başına maksimum önyükleme girişimi sayısı. İlk turdan sonraki her tur, önbellekleri atlamak ve çeşitli izler toplamak için `temperature=1.0` ile yeni bir çalıştırıcı (rollout) kullanır. Herhangi bir turda başarılı bir önyükleme bulunursa, örnek kabul edilir ve optimize edici bir sonrakine geçer. Varsayılan 1'dir. | `1` |
| `max_errors` | `Optional[int]` | Program sonlanana kadar tolere edilecek maksimum hata sayısı. Eğer `None` ise, `dspy.settings.max_errors` değerini miras alır. | `None` |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap.py`*

---

## Functions (Fonksiyonlar)

### `compile`

```python
compile(student, *, teacher=None, trainset)
```

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/bootstrap.py`*

### `get_params`

```python
get_params() -> dict[str, Any]
```

Teleprompter'ın parametrelerini alır.

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `dict[str, Any]` | Teleprompter'ın parametreleri. |

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/teleprompt/teleprompt.py`*
