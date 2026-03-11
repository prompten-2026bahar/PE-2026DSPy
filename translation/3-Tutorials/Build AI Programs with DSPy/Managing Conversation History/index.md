# Konuşma Geçmişini Yönetme

Konuşma geçmişini korumak, sohbet robotları gibi yapay zeka uygulamaları oluştururken temel bir özelliktir. DSPy, `dspy.Module` içinde otomatik konuşma geçmişi yönetimi sağlamasa da, konuşma geçmişini etkili bir şekilde yönetmenize yardımcı olmak için `dspy.History` yardımcı aracını sunar.

## Konuşma Geçmişini Yönetmek İçin `dspy.History` Kullanımı

`dspy.History` sınıfı, konuşma geçmişini saklayan bir `messages: list[dict[str, Any]]` niteliği içeren bir giriş alanı (input field) türü olarak kullanılabilir. Bu listedeki her girdi, imzanızda (signature) tanımlanan alanlara karşılık gelen anahtarları içeren bir sözlüktür. Aşağıdaki örneği inceleyin:

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
    question = input("Type your question, end conversation by typing 'finish': ")
    if question == "finish":
        break
    outputs = predict(question=question, history=history)
    print(f"\n{outputs.answer}\n")
    history.messages.append({"question": question, **outputs})

dspy.inspect_history()
```

Konuşma geçmişini kullanırken iki temel adım vardır:

- **İmzanıza (Signature) `dspy.History` türünde bir alan dahil edin.**
- **Çalışma zamanında bir geçmiş örneği (instance) tutun ve yeni konuşma turlarını buna ekleyin.** Her girdi, ilgili tüm giriş ve çıkış alanı bilgilerini içermelidir.

Örnek bir çalışma şu şekilde görünebilir:

```
Type your question, end conversation by typing 'finish': do you know the competition between pytorch and tensorflow?

Yes, there is a notable competition between PyTorch and TensorFlow, which are two of the most popular deep learning frameworks. PyTorch, developed by Facebook, is known for its dynamic computation graph, which allows for more flexibility and ease of use, especially in research settings. TensorFlow, developed by Google, initially used a static computation graph but has since introduced eager execution to improve usability. TensorFlow is often favored in production environments due to its scalability and deployment capabilities. Both frameworks have strong communities and extensive libraries, and the choice between them often depends on specific project requirements and personal preference.

Type your question, end conversation by typing 'finish': which one won the battle? just tell me the result, don't include any reasoning, thanks!

There is no definitive winner; both PyTorch and TensorFlow are widely used and have their own strengths.
Type your question, end conversation by typing 'finish': finish




[2025-07-11T16:35:57.592762]

System message:

Your input fields are:
1. `question` (str): 
2. `history` (History):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, `history`, produce the fields `answer`.


User message:

[[ ## question ## ]]
do you know the competition between pytorch and tensorflow?


Assistant message:

[[ ## answer ## ]]
Yes, there is a notable competition between PyTorch and TensorFlow, which are two of the most popular deep learning frameworks. PyTorch, developed by Facebook, is known for its dynamic computation graph, which allows for more flexibility and ease of use, especially in research settings. TensorFlow, developed by Google, initially used a static computation graph but has since introduced eager execution to improve usability. TensorFlow is often favored in production environments due to its scalability and deployment capabilities. Both frameworks have strong communities and extensive libraries, and the choice between them often depends on specific project requirements and personal preference.

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
which one won the battle? just tell me the result, don't include any reasoning, thanks!

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## answer ## ]]
There is no definitive winner; both PyTorch and TensorFlow are widely used and have their own strengths.

[[ ## completed ## ]]
```

Kullanıcı girişlerinin ve asistan yanıtlarının geçmişe nasıl eklendiğine dikkat edin; bu, modelin turlar boyunca bağlamı korumasını sağlar.

Dil modeline gönderilen asıl istem (prompt), `dspy.inspect_history` çıktısında gösterildiği gibi çok turlu bir mesajdır. Her konuşma turu, bir kullanıcı mesajı ve ardından gelen bir asistan mesajı olarak temsil edilir.

## Az-Örnekli (Few-shot) Örneklerde Geçmiş

`history` alanı giriş alanları arasında listelenmiş olsa bile (örneğin, sistem mesajında "2. `history` (History):" şeklinde), istemin giriş alanları bölümünde görünmediğini fark edebilirsiniz. Bu kasıtlıdır: Konuşma geçmişi içeren az-örnekli örnekler formatlanırken, DSPy geçmişi birden fazla tura genişletmez. Bunun yerine, OpenAI standart formatıyla uyumlu kalmak için her az-örnekli örnek tek bir tur olarak temsil edilir.

Örneğin:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()


predict = dspy.Predict(QA)
history = dspy.History(messages=[])

predict.demos.append(
    dspy.Example(
        question="What is the capital of France?",
        history=dspy.History(
            messages=[{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]
        ),
        answer="The capital of France is Paris.",
    )
)

predict(question="What is the capital of America?", history=dspy.History(messages=[]))
dspy.inspect_history()
```

Sonuçta oluşan geçmiş şu şekilde görünecektir:

```
[2025-07-11T16:53:10.994111]

System message:

Your input fields are:
1. `question` (str): 
2. `history` (History):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## history ## ]]
{history}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, `history`, produce the fields `answer`.


User message:

[[ ## question ## ]]
What is the capital of France?

[[ ## history ## ]]
{"messages": [{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]}


Assistant message:

[[ ## answer ## ]]
The capital of France is Paris.

[[ ## completed ## ]]


User message:

[[ ## question ## ]]
What is the capital of America?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## answer ## ]]
The capital of the United States of America is Washington, D.C.

[[ ## completed ## ]]
```

Gördüğünüz gibi, az-örnekli (few-shot) örnek, konuşma geçmişini birden fazla tura genişletmez. Bunun yerine, geçmişi kendi bölümü içinde JSON verisi olarak temsil eder::

```
[[ ## history ## ]]
{"messages": [{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]}
```

Bu yaklaşım, standart istem formatlarıyla uyumluluğu sağlarken modelin ilgili konuşma bağlamına erişimini korumasına olanak tanır.
