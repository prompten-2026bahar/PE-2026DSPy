# Eğitim: Gelişmiş Araç Kullanımı

Gelişmiş araç kullanımı için bir DSPy ajanı oluşturup istem-optimizasyonu yapmaya yönelik hızlı bir örneği adım adım inceleyelim. Bunu zorlu [ToolHop](https://arxiv.org/abs/2501.02506) görevi için yapacağız, ancak daha da katı bir değerlendirme ölçütü kullanacağız.

En güncel DSPy sürümünü `pip install -U dspy` ile kurup devam edin. Ayrıca `pip install func_timeout datasets` komutunu da çalıştırmanız gerekir.

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. MLflow’u aşağıdaki dört adımı izleyerek kolayca kurabilirsiniz.

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow arayüzünü başlatın
```bash
mlflow ui --port 5000
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.
</details>

Bu eğitimde, daha büyük LLM’ler ve daha zor görevler için güçlü olma eğiliminde olan yeni deneysel `dspy.SIMBA` istem optimize edicisini göstereceğiz. Bunu kullanarak ajanımızı %35 doğruluktan %60 doğruluğa çıkaracağız.

```python
import dspy
import orjson
import random

gpt4o = dspy.LM("openai/gpt-4o", temperature=0.7)
dspy.configure(lm=gpt4o)
```

Şimdi veriyi indirelim.

```python
from dspy.utils import download

download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json")

data = orjson.loads(open("ToolHop.json").read())
random.Random(0).shuffle(data)
```

Ardından temizlenmiş bir örnek kümesi hazırlayalım. ToolHop görevi şu açıdan ilginçtir: ajan, her istek için ayrı ayrı kullanacağı _benzersiz bir_ araç (fonksiyon) kümesi alır. Dolayısıyla pratikte bu tür araçların _herhangi birini_ etkili şekilde kullanmayı öğrenmesi gerekir.

```python
import re
import inspect

examples = []
fns2code = {}

def finish(answer: str):
    """İzleği sonlandır ve nihai cevabı döndür."""
    return answer

for datapoint in data:
    func_dict = {}
    for func_code in datapoint["functions"]:
        cleaned_code = func_code.rsplit("\n\n# Example usage", 1)[0]
        fn_name = re.search(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", cleaned_code)
        fn_name = fn_name.group(1) if fn_name else None

        if not fn_name:
            continue

        local_vars = {}
        exec(cleaned_code, {}, local_vars)
        fn_obj = local_vars.get(fn_name)

        if callable(fn_obj):
            func_dict[fn_name] = fn_obj
            assert fn_obj not in fns2code, f"Yinelenen fonksiyon bulundu: {fn_name}"
            fns2code[fn_obj] = (fn_name, cleaned_code)

    func_dict["finish"] = finish

    example = dspy.Example(question=datapoint["question"], answer=datapoint["answer"], functions=func_dict)
    examples.append(example.with_inputs("question", "functions"))

trainset, devset, testset = examples[:100], examples[100:400], examples[400:]
```

Şimdi görev için bazı yardımcılar tanımlayalım. Burada `metric` tanımlayacağız; bu, orijinal makaledekinden (çok) daha katı olacak: tahminin normalize edildikten sonra tam olarak gerçek yanıtla eşleşmesini bekleyeceğiz. İkinci bir açıdan da katı olacağız: verimli dağıtımı mümkün kılmak için ajanın toplamda yalnızca 5 adım atmasına izin vereceğiz.

```python
from func_timeout import func_set_timeout

def wrap_function_with_timeout(fn):
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return {"return_value": fn(*args, **kwargs), "errors": None}
        except Exception as e:
            return {"return_value": None, "errors": str(e)}

    return wrapper

def fn_metadata(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "Docstring yok."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)

def metric(example, pred, trace=None):
    gold = str(example.answer).rstrip(".0").replace(",", "").lower()
    pred = str(pred.answer).rstrip(".0").replace(",", "").lower()
    return pred == gold  # orijinal makalenin metriğinden daha katı!

evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=0, max_errors=999)
```

Şimdi ajanı tanımlayalım! Ajanımızın çekirdeği bir ReAct döngüsüne dayanacak; bu döngüde model şimdiye kadarki izleği ve çağırabileceği fonksiyonlar kümesini görür ve bir sonraki çağıracağı araca karar verir.

Nihai ajanı hızlı tutmak için `max_steps` değerini 5 adımla sınırlandıracağız. Ayrıca her fonksiyon çağrısını bir zaman aşımı ile çalıştıracağız.

```python
class Agent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        instructions = "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, names as Firstname Lastname, and numbers without leading 0s."
        signature = dspy.Signature('question, trajectory, functions -> next_selected_fn, args: dict[str, Any]', instructions)
        self.react = dspy.ChainOfThought(signature)

    def forward(self, question, functions):
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []

        for _ in range(self.max_steps):
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            selected_fn = pred.next_selected_fn.strip('"').strip("'")
            fn_output = wrap_function_with_timeout(functions[selected_fn])(**pred.args)
            trajectory.append(dict(reasoning=pred.reasoning, selected_fn=selected_fn, args=pred.args, **fn_output))

            if selected_fn == "finish":
                break

        return dspy.Prediction(answer=fn_output.get("return_value", ''), trajectory=trajectory)
```

Kutudan çıktığı haliyle, `GPT-4o` destekli ajanımızı geliştirme kümesi üzerinde değerlendirelim.

```python
agent = Agent()
evaluate(agent)
```

Şimdi ajanı **Stochastic Introspective Mini-Batch Ascent** anlamına gelen `dspy.SIMBA` ile optimize edelim. Bu istem optimize edici, burada kullandığımız ajan gibi rastgele DSPy programlarını kabul eder ve istem talimatlarında veya few-shot örneklerde kademeli iyileştirmeler arayan mini-batch dizileri halinde ilerler.

```python
simba = dspy.SIMBA(metric=metric, max_steps=12, max_demos=10)
optimized_agent = simba.compile(agent, trainset=trainset, seed=6793115)
```

Bu optimizasyon tamamlandıktan sonra, şimdi ajanımızı yeniden değerlendirelim. %71’lik göreli bir artışla doğruluğun %60’a sıçradığını görüyoruz.

```python
evaluate(optimized_agent)
```
