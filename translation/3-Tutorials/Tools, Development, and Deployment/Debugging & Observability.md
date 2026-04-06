# Eğitim: DSPy’de Hata Ayıklama ve Gözlemlenebilirlik

Bu kılavuz, DSPy’de sorunları nasıl ayıklayacağınızı ve gözlemlenebilirliği nasıl geliştireceğinizi gösterir. Modern yapay zekâ programları çoğu zaman dil modelleri, getirme bileşenleri ve araçlar gibi birden fazla parçadan oluşur. DSPy, bu tür karmaşık yapay zekâ sistemlerini temiz ve modüler bir şekilde kurmanıza ve optimize etmenize olanak tanır.

Ancak sistemler daha karmaşık hâle geldikçe, **sisteminizin ne yaptığını anlayabilme** yeteneği kritik hâle gelir. Şeffaflık olmadan tahmin süreci kolayca bir kara kutuya dönüşebilir; bu da hataları veya kalite sorunlarını teşhis etmeyi zorlaştırır ve üretimde bakım yapmayı güçleştirir.

Bu eğitimin sonunda, bir sorunu nasıl ayıklayacağınızı ve [MLflow Tracing](#tracing) kullanarak gözlemlenebilirliği nasıl geliştireceğinizi anlayacaksınız. Ayrıca callback’ler kullanarak özel bir loglama çözümünü nasıl oluşturabileceğinizi de inceleyeceksiniz.

## Bir Program Tanımlama

Basit bir ReAct ajanı oluşturarak başlayacağız; bu ajan getirme kaynağı olarak ColBERTv2’nin Wikipedia veri kümesini kullanır. Bunu daha gelişmiş bir programla değiştirebilirsiniz.

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """ColBert içinden en alakalı ilk 3 bilgiyi getir"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
```

Şimdi ajana basit bir soru soralım:

```python
prediction = agent(question="Which baseball team does Shohei Ohtani play for in June 2025?")
print(prediction.answer)
```

```
Shohei Ohtani is expected to play for the Hokkaido Nippon-Ham Fighters in June 2025, based on the available information.
```

Bu yanlış. Artık Hokkaido Nippon-Ham Fighters için oynamıyor; Dodgers’a geçti ve 2024’te World Series kazandı! Programı ayıklayalım ve olası düzeltmeleri inceleyelim.

## `inspect_history` Kullanımı

DSPy, şu ana kadar yapılan tüm LLM çağrılarını yazdıran `inspect_history()` yardımcı işlevini sağlar:

```python
# 5 LLM çağrısını yazdır
dspy.inspect_history(n=5)
```

```
[2024-12-01T10:23:29.144257]

System message:

Your input fields are:
1. `question` (str)

...

Response:

Response:

[[ ## reasoning ## ]]
The search for information regarding Shohei Ohtani's team in June 2025 did not yield any specific results. The retrieved data consistently mentioned that he plays for the Hokkaido Nippon-Ham Fighters, but there was no indication of any changes or updates regarding his team for the specified date. Given the lack of information, it is reasonable to conclude that he may still be with the Hokkaido Nippon-Ham Fighters unless there are future developments that are not captured in the current data.

[[ ## answer ## ]]
Shohei Ohtani is expected to play for the Hokkaido Nippon-Ham Fighters in June 2025, based on the available information.

[[ ## completed ## ]]
```

Log, ajanın arama aracından faydalı bilgi getiremediğini gösteriyor. Peki getirici tam olarak ne döndürdü? Faydalı olsa da, `inspect_history`’nin bazı sınırlamaları vardır:

* Gerçek dünya sistemlerinde getiriciler, araçlar ve özel modüller gibi başka bileşenler de önemli rol oynar; ancak `inspect_history` yalnızca LLM çağrılarını loglar.
* DSPy programları çoğu zaman tek bir tahmin içinde birden fazla LLM çağrısı yapar. Tek parça log geçmişi, özellikle birden fazla soru ele alınırken logları düzenlemeyi zorlaştırır.
* Parametreler, gecikme süresi ve modüller arasındaki ilişki gibi üst veriler yakalanmaz.

**Tracing**, bu sınırlamaları giderir ve daha kapsamlı bir çözüm sunar.

## Tracing

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html), LLMOps için en iyi uygulamaları desteklemek üzere DSPy ile sorunsuz biçimde entegre olan uçtan uca bir makine öğrenmesi platformudur. DSPy ile MLflow’un otomatik tracing yeteneğini kullanmak oldukça kolaydır; **hiçbir hizmete kayıt olmanız veya API anahtarı almanız gerekmez**. Sadece MLflow’u kurmanız ve not defterinizde veya betiğinizde `mlflow.dspy.autolog()` çağırmanız yeterlidir.

```bash
pip install -U mlflow>=2.18.0
```

Kurulumdan sonra, aşağıdaki komutla sunucunuzu başlatın.

```
# MLflow tracing kullanırken SQL store kullanmanız şiddetle tavsiye edilir
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

Eğer `--port` bayrağıyla farklı bir port belirtmezseniz, MLflow sunucunuz 5000 portunda çalışacaktır.

Şimdi MLflow tracing’i etkinleştirmek için kodumuzu değiştirelim. Şunları yapmamız gerekir:

- MLflow’a sunucunun nerede barındırıldığını söylemek.
- DSPy tracing’in otomatik olarak yakalanması için `mlflow.autolog()` uygulamak.

Tam kod aşağıdadır; şimdi tekrar çalıştıralım!

```python
import dspy
import os
import mlflow

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# MLflow'a sunucu URI'sini bildir.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Deneyiniz için benzersiz bir ad oluşturun.
mlflow.set_experiment("DSPy")

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """ColBert içinden en alakalı ilk 3 bilgiyi getir"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
print(agent(question="Which baseball team does Shohei Ohtani play for?"))
```

MLflow, her tahmin için otomatik olarak bir **trace** üretir ve bunu deneyiniz içinde kaydeder. Bu trace’leri görsel olarak incelemek için tarayıcınızda `http://127.0.0.1:5000/` adresini açın, ardından az önce oluşturduğunuz deneyi seçin ve Traces sekmesine gidin:

![MLflow Trace UI](./mlflow_trace_ui.png)

En son trace’e tıklayarak ayrıntılı kırılımını görüntüleyin:

![MLflow Trace View](./mlflow_trace_view.png)

Burada iş akışınızdaki her adımın girdi ve çıktısını inceleyebilirsiniz. Örneğin yukarıdaki ekran görüntüsü `retrieve` fonksiyonunun girdisini ve çıktısını gösterir. Getiricinin çıktısını inceleyerek eski bilgi döndürdüğünü ve bunun Shohei Ohtani’nin Haziran 2025’te hangi takımda oynadığını belirlemek için yeterli olmadığını görebilirsiniz. Ayrıca başka adımları da inceleyebilirsiniz; örneğin dil modelinin girdisi, çıktısı ve yapılandırması.

Güncelliğini yitirmiş bilgi sorununu çözmek için `retrieve` fonksiyonunu, [Tavily search](https://www.tavily.com/) ile desteklenen bir web arama aracıyla değiştirebilirsiniz.

```python
from tavily import TavilyClient
import dspy
import mlflow

# MLflow'a sunucu URI'sini bildir.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Deneyiniz için benzersiz bir ad oluşturun.
mlflow.set_experiment("DSPy")

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """Web araması yap ve ilk 5 sonucun içeriğini döndür"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[web_search])

prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(agent.answer)
```

```
Los Angeles Dodgers
```

Aşağıda MLflow arayüzünde nasıl gezinileceğini gösteren bir GIF bulunmaktadır:

![MLflow Trace UI Navigation](./mlflow_trace_ui_navigation.gif)

MLflow tracing’in nasıl kullanılacağına dair tam kılavuz için lütfen [MLflow Tracing Guide](https://mlflow.org/docs/3.0.0rc0/tracing) sayfasına bakın.

!!! info "MLflow hakkında daha fazlasını öğrenin"

    MLflow; deney takibi, değerlendirme ve dağıtım gibi kapsamlı özellikler sunan uçtan uca bir LLMOps platformudur. DSPy ve MLflow entegrasyonu hakkında daha fazla bilgi için [bu eğitime](../deployment/index.md#deploying-with-mlflow) göz atın.

## Özel Bir Loglama Çözümü Oluşturma

Bazen özel bir loglama çözümü uygulamak isteyebilirsiniz. Örneğin, belirli bir modül tarafından tetiklenen özel olayları loglamanız gerekebilir. DSPy’nin **callback** mekanizması bu tür kullanım senaryolarını destekler. `BaseCallback` sınıfı, loglama davranışını özelleştirmek için çeşitli handler’lar sağlar:

|Handlers|Description|
|:--|:--|
|`on_module_start` / `on_module_end` | Bir `dspy.Module` alt sınıfı çağrıldığında tetiklenir. |
|`on_lm_start` / `on_lm_end` | Bir `dspy.LM` alt sınıfı çağrıldığında tetiklenir. |
|`on_adapter_format_start` / `on_adapter_format_end`| Bir `dspy.Adapter` alt sınıfı giriş istemini biçimlendirdiğinde tetiklenir. |
|`on_adapter_parse_start` / `on_adapter_parse_end`| Bir `dspy.Adapter` alt sınıfı, bir LM’den gelen çıktı metnini sonradan işlediğinde tetiklenir. |
|`on_tool_start` / `on_tool_end` | Bir `dspy.Tool` alt sınıfı çağrıldığında tetiklenir. |
|`on_evaluate_start` / `on_evaluate_end` | Bir `dspy.Evaluate` örneği çağrıldığında tetiklenir. |

Aşağıda, bir ReAct ajanının ara adımlarını loglayan özel bir callback örneği verilmiştir:

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. BaseCallback sınıfını genişleten özel bir callback sınıfı tanımla
class AgentLoggingCallback(BaseCallback):

    # 2. Özel loglama kodu çalıştırmak için on_module_end handler'ını uygula.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Callback'i DSPy ayarına ekle; böylece program çalıştırmalarına uygulanır
dspy.configure(callbacks=[AgentLoggingCallback()])
```

```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Ohtani ..."]

...
```

!!! info "Callback’lerde Girdi ve Çıktılarla Çalışma"

    Callback’lerde girdi veya çıktı verileriyle çalışırken dikkatli olun. Bunları yerinde değiştirmek, programa geçen orijinal veriyi de değiştirebilir ve beklenmedik davranışlara yol açabilir. Bunu önlemek için, veriyi değiştirebilecek herhangi bir işlem yapmadan önce verinin bir kopyasını oluşturmanız kuvvetle tavsiye edilir.
