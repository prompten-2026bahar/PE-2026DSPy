# /translation/history.md

## dspy.History

```python
dspy.History
```

**Kullanılan Yapılar (Bases):** `BaseModel`

Konuşma geçmişini (conversation history) temsil eden sınıf.

Konuşma geçmişi bir mesaj listesidir, her mesaj varlığı (entity) ilişkili imzadan (signature) gelen anahtarlara sahip olmalıdır. Örneğin, aşağıdaki imzaya sahipseniz:

```python
class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()
```

O zaman geçmiş (history), `"question"` ve `"answer"` anahtarlarına sahip sözlüklerden (dictionaries) oluşan bir liste olmalıdır.

**Örnekler (Examples):**

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

history = dspy.History(
    messages=[
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
    ]
)

predict = dspy.Predict(MySignature)
outputs = predict(question="What is the capital of France?", history=history)
```

Konuşma geçmişini yakalama (capturing) örneği:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(MySignature)
outputs = predict(question="What is the capital of France?")

history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
outputs_with_history = predict(question="Are you sure?", history=history)
```

---

## Öznitelikler (Attributes)

### `messages`

```python
messages: list[dict[str, Any]] # instance-attribute
```

### `model_config`

```python
model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, validate_assignment=True, extra='forbid') # class-attribute instance-attribute
```
