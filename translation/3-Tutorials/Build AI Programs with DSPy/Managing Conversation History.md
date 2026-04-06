# Sohbet Geçmişini Yönetme (Managing Conversation History)

Sohbet geçmişini korumak, sohbet botları (chatbots) gibi yapay zeka uygulamaları oluştururken temel bir özelliktir. DSPy, `dspy.Module` içinde otomatik bir sohbet geçmişi yönetimi sağlamasa da, bu süreci etkili bir şekilde yönetmenize yardımcı olmak için `dspy.History` aracını (utility) sunar.

## Sohbet Geçmişini Yönetmek İçin `dspy.History` Kullanımı

`dspy.History` sınıfı, bir girdi alanı (input field) türü olarak kullanılabilir. Sohbet geçmişini depolayan `messages: list[dict[str, Any]]` niteliğine sahiptir. Bu listedeki her bir kayıt, imzanızda (signature) tanımlanan alanlara karşılık gelen anahtarları (keys) içeren bir sözlüktür (dictionary). 

Aşağıdaki örneğe göz atabilirsiniz:

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{kendi_openai_api_anahtariniz}"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
    question = input("Sorunuzu yazın, sohbeti sonlandırmak için 'finish' yazın: ")
    if question == "finish":
        break
        
    outputs = predict(question=question, history=history)
    print(f"\n{outputs.answer}\n")
    
    # Yeni turu geçmişe ekliyoruz
    history.messages.append({"question": question, **outputs})

dspy.inspect_history()
```

Sohbet geçmişini kullanırken dikkat etmeniz gereken **iki önemli adım** vardır:
* İmzanıza (Signature) `dspy.History` türünde bir alan ekleyin.
* Çalışma zamanında (runtime) bir geçmiş nesnesi (history instance) oluşturun ve yeni sohbet turlarını (turns) buraya ekleyerek güncel tutun. Eklenen her kayıt, ilgili tüm girdi ve çıktı alanı bilgilerini içermelidir.

Örnek bir çalışma ekranı şuna benzeyebilir:

> **Sorunuzu yazın, sohbeti sonlandırmak için 'finish' yazın:** pytorch ve tensorflow arasındaki rekabeti biliyor musun?
> 
> *Evet, en popüler iki derin öğrenme altyapısı olan PyTorch ve TensorFlow arasında belirgin bir rekabet var. Facebook tarafından geliştirilen PyTorch, özellikle araştırma ortamlarında daha fazla esneklik ve kullanım kolaylığı sağlayan dinamik hesaplama grafiğiyle bilinir. TensorFlow ise...*
>
> **Sorunuzu yazın, sohbeti sonlandırmak için 'finish' yazın:** hangisi savaşı kazandı? bana sadece sonucu söyle, mantıksal bir açıklama yapma, teşekkürler!
>
> *Kesin bir kazanan yok; hem PyTorch hem de TensorFlow yaygın olarak kullanılıyor ve kendi güçlü yönleri var.*

Modelin, farklı sohbet turları arasındaki bağlamı koruyabilmesi için her kullanıcı girdisinin ve asistan yanıtının geçmişe (history) nasıl eklendiğine dikkat edin. Dil modeline gönderilen asıl komut (prompt), `dspy.inspect_history()` çıktısında da görebileceğiniz gibi çok turlu (multi-turn) bir mesajdır. Her sohbet turu, önce bir kullanıcı mesajı, ardından gelen bir asistan mesajı olarak temsil edilir:

```text
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

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.

Assistant message:
[[ ## answer ## ]]
Yes, there is a notable competition between PyTorch and TensorFlow...

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

---

## Few-shot (Az Örnekli) Örneklerde Geçmişin Durumu

Geçmiş (history) bir girdi alanı olarak tanımlanmasına rağmen (örneğin, sistem mesajında *"2. history (History):"* şeklinde listelenir), prompt'un asıl girdi alanları bölümünde görünmediğini fark edebilirsiniz. 

Bu durum tamamen **kasıtlıdır**: DSPy, sohbet geçmişi içeren few-shot (az örnekli) örnekleri formatlarken geçmişi birden çok tura genişletmez (expand etmez). Bunun yerine, OpenAI'ın standart formatıyla uyumlu kalabilmek için her bir few-shot örneğini **tek bir tur** olarak temsil eder.

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
        question="Fransa'nın başkenti neresidir?",
        history=dspy.History(
            messages=[{"question": "Almanya'nın başkenti neresidir?", "answer": "Almanya'nın başkenti Berlin'dir."}]
        ),
        answer="Fransa'nın başkenti Paris'tir.",
    )
)

predict(question="Amerika'nın başkenti neresidir?", history=dspy.History(messages=[]))
dspy.inspect_history()
```

Bunun sonucunda oluşan geçmiş (history) çıktısı şuna benzeyecektir:

```text
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
Fransa'nın başkenti neresidir?

[[ ## history ## ]]
{"messages": [{"question": "Almanya'nın başkenti neresidir?", "answer": "Almanya'nın başkenti Berlin'dir."}]}

Assistant message:
[[ ## answer ## ]]
Fransa'nın başkenti Paris'tir.

[[ ## completed ## ]]

User message:
[[ ## question ## ]]
Amerika'nın başkenti neresidir?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.

Response:
[[ ## answer ## ]]
Amerika Birleşik Devletleri'nin başkenti Washington, D.C.'dir.

[[ ## completed ## ]]
```

Görebileceğiniz gibi, few-shot (az örnekli) kullanım, sohbet geçmişini çoklu diyalog turları halinde yaymaz. Bunun yerine, geçmişi kendi bölümünde bir **JSON verisi** olarak temsil eder:

```json
[[ ## history ## ]]
{"messages": [{"question": "Almanya'nın başkenti neresidir?", "answer": "Almanya'nın başkenti Berlin'dir."}]}
```

Bu yaklaşım, dil modeline gerekli sohbet bağlamını başarıyla aktarırken aynı zamanda standart prompt formatlarıyla tam uyumluluk sağlar.