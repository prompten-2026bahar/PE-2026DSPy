# DSPy Adaptörlerini Anlamak

## Adaptörler Nedir?

Adaptörler, `dspy.Predict` ile gerçek Dil Modeli (LM) arasındaki köprüdür. Bir DSPy modülünü çağırdığınızda, adaptör; imzanızı, kullanıcı girdilerini ve `demos` (az-örnekli gösterimler) gibi diğer öznitelikleri alır ve bunları LM'ye gönderilen çok turlu mesajlara dönüştürür.

Adaptör sistemi şunlardan sorumludur:

- DSPy imzalarını, görevi ve istek/yanıt yapısını tanımlayan sistem mesajlarına çevirmek.
- Girdi verilerini, DSPy imzalarında ana hatları belirtilen istek yapısına göre formatlamak.
- LM yanıtlarını tekrar `dspy.Prediction` örnekleri gibi yapılandırılmış DSPy çıktılarına ayrıştırmak (parsing).
- Konuşma geçmişini ve fonksiyon çağrılarını yönetmek.
- `dspy.Tool`, `dspy.Image` vb. gibi önceden oluşturulmuş DSPy türlerini LM istem (prompt) mesajlarına dönüştürmek.

## Adaptörleri Yapılandırma

Tüm Python süreci için adaptör seçmek üzere `dspy.configure(adapter=...)` kullanabilir veya yalnızca belirli bir isim alanını (namespace) etkilemek için `with dspy.context(adapter=...):` yapısını kullanabilirsiniz.

Eğer DSPy iş akışında herhangi bir adaptör belirtilmemişse, her `dspy.Predict.__call__` varsayılan olarak `dspy.ChatAdapter` kullanır. Bu nedenle, aşağıdaki iki kod parçacığı birbirine eşdeğerdir:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
```

```python
import dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(),  # This is the default value
)

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
```

## Adaptörler Sistemin Neresinde Yer Alır?

Akış şu şekilde işler:

1. Kullanıcı, genellikle girdileri olan bir `dspy.Module` olan kendi DSPy ajanını çağırır.
2. LM yanıtını almak için dahili `dspy.Predict` tetiklenir.
3. `dspy.Predict`, imzayı, girdileri ve gösterimleri (demos) `dspy.LM`'ye gönderilen çok turlu mesajlara dönüştüren **Adapter.format()** metodunu çağırır. `dspy.LM`, LM uç noktasıyla iletişim kuran `litellm` etrafındaki ince bir sarmalayıcıdır.
4. LM mesajları alır ve bir yanıt oluşturur.
5. **Adapter.parse()**, imzada belirtildiği gibi LM yanıtını yapılandırılmış DSPy çıktılarına dönüştürür.
6. `dspy.Predict` çağırıcısı, ayrıştırılmış çıktıları alır.

LM'ye gönderilen mesajları görüntülemek için `Adapter.format()` metodunu açıkça çağırabilirsiniz.

```python
# Basitleştirilmiş akış örneği
signature = dspy.Signature("question -> answer")
inputs = {"question": "What is 2+2?"}
demos = [{"question": "What is 1+1?", "answer": "2"}]

adapter = dspy.ChatAdapter()
print(adapter.format(signature, demos, inputs))
```

Çıktı şuna benzeyecektir:

```
{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 1+1?'}
{'role': 'assistant', 'content': '[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 2+2?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}
```

Ayrıca, yalnızca `adapter.format_system_message(signature)` metodunu çağırarak sistem mesajını alabilirsiniz.

```python
import dspy

signature = dspy.Signature("question -> answer")
system_message = dspy.ChatAdapter().format_system_message(signature)
print(system_message)
```

Çıktı şuna benzeyecektir:

```
Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}
[[ ## answer ## ]]
{answer}
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, produce the fields `answer`.
```

## Adaptör Türleri

DSPy, her biri belirli kullanım durumlarına göre tasarlanmış birkaç adaptör türü sunar:

### ChatAdapter

**ChatAdapter** varsayılan adaptördür ve tüm dil modelleriyle çalışır. Özel işaretçilere sahip alan tabanlı (field-based) bir format kullanır.

#### Format Yapısı

ChatAdapter, alanları birbirinden ayırmak için `[[ ## alan_adi ## ]]` işaretçilerini kullanır. İlkel olmayan Python türündeki alanlar için, ilgili türün JSON şemasını da dahil eder. Aşağıda, `dspy.ChatAdapter` tarafından formatlanan mesajları net bir şekilde görüntülemek için `dspy.inspect_history()` metodunu kullanıyoruz.

```python
import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.ChatAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """Get news about the given science field"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="science news")


predict = dspy.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
dspy.inspect_history()
```

Çıktı şu şekilde görünür::

```
[2025-08-15T22:24:29.378666]

System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

[[ ## news ## ]]
{news}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "$defs": {"ScienceNews": {"type": "object", "properties": {"scientists_involved": {"type": "array", "items": {"type": "string"}, "title": "Scientists Involved"}, "text": {"type": "string", "title": "Text"}}, "required": ["text", "scientists_involved"], "title": "ScienceNews"}}, "items": {"$ref": "#/$defs/ScienceNews"}}

[[ ## completed ## ]]
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with the corresponding output fields, starting with the field `[[ ## news ## ]]` (must be formatted as a valid Python list[ScienceNews]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## news ## ]]
[
    {
        "scientists_involved": ["John Doe", "Jane Smith"],
        "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating their potential to solve complex problems faster than classical computers. This breakthrough could revolutionize fields such as cryptography and optimization."
    }
]

[[ ## completed ## ]]
```

!!! info "Pratik: Yazdırılan LM geçmişinde İmza (Signature) bilgilerini bulun"
    İmzayı değiştirmeyi deneyin ve değişikliklerin yazdırılan LM mesajına nasıl yansıdığını gözlemleyin.


Her alanın başında `[[ ## alan_adi ## ]]` şeklinde bir işaretçi bulunur. Eğer bir çıktı alanı ilkel olmayan (non-primitive) türlere sahipse, talimat ilgili türün JSON şemasını içerir ve çıktı buna göre formatlanır. Çıktı alanı `ChatAdapter` tarafından tanımlanan şekilde yapılandırıldığı için, otomatik olarak yapılandırılmış veriye ayrıştırılabilir.

#### ChatAdapter Ne Zaman Kullanılmalı?

`ChatAdapter` aşağıdaki avantajları sunar:

- **Evrensel uyumluluk**: Tüm dil modelleriyle çalışır; ancak daha küçük modeller istenen formata uymayan yanıtlar üretebilir.
- **Geri dönüş koruması (Fallback)**: Eğer `ChatAdapter` başarısız olursa, otomatik olarak `JSONAdapter` ile yeniden dener.

Genel olarak, özel gereksinimleriniz yoksa `ChatAdapter` güvenilir bir seçimdir.

#### ChatAdapter Ne Zaman Kullanılmamalı?

Şu durumlarda `ChatAdapter` kullanmaktan kaçının:

- **Gecikme (Latency) hassasiyeti**: `ChatAdapter`, diğer adaptörlere kıyasla daha fazla "kalıp" (boilerplate) çıktı token'ı içerir. Gecikmeye duyarlı bir sistem kuruyorsanız farklı bir adaptör kullanmayı düşünebilirsiniz.

### JSONAdapter

**JSONAdapter**, LM'yi imzada belirtilen tüm çıktı alanlarını içeren JSON verisi döndürmeye zorlar. `response_format` parametresi aracılığıyla yapılandırılmış çıktıyı destekleyen modeller için etkilidir; daha güvenilir ayrıştırma için yerel JSON üretim yeteneklerinden yararlanır.

#### Format Yapısı

`JSONAdapter` tarafından formatlanan istemin (prompt) girdi kısmı `ChatAdapter`'a benzer, ancak çıktı kısmı aşağıda gösterildiği gibi farklılık gösterir:

```python
import dspy
import pydantic

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=dspy.JSONAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(dspy.Signature):
    """Get news about the given science field"""

    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="science news")


predict = dspy.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
dspy.inspect_history()
```

```
System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

Outputs will be a JSON object with the following fields.

{
  "news": "{news}        # note: the value you produce must adhere to the JSON schema: {\"type\": \"array\", \"$defs\": {\"ScienceNews\": {\"type\": \"object\", \"properties\": {\"scientists_involved\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"title\": \"Scientists Involved\"}, \"text\": {\"type\": \"string\", \"title\": \"Text\"}}, \"required\": [\"text\", \"scientists_involved\"], \"title\": \"ScienceNews\"}}, \"items\": {\"$ref\": \"#/$defs/ScienceNews\"}}"
}
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with a JSON object in the following order of fields: `news` (must be formatted as a valid Python list[ScienceNews]).


Response:

{
  "news": [
    {
      "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating that quantum systems can outperform classical computers in specific tasks. This breakthrough could revolutionize fields such as cryptography and complex system simulations.",
      "scientists_involved": [
        "Dr. Alice Smith",
        "Dr. Bob Johnson",
        "Dr. Carol Lee"
      ]
    }
  ]
}
```

#### When to Use JSONAdapter

`JSONAdapter` is good at:

- **Structured output support**: When the model supports the `response_format` parameter.
- **Low latency**: Minimal boilerplate in the LM response results in faster responses.

#### When Not to Use JSONAdapter

Avoid using `JSONAdapter` if you are:

- Using a model that does not natively support structured output, such as a small open-source model hosted on Ollama.

## Summary

Adapters are a crucial component of DSPy that bridge the gap between structured DSPy signatures and language model APIs.
Understanding when and how to use different adapters will help you build more reliable and efficient DSPy programs.
