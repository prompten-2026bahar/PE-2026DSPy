# Eğitim: DSPy Programlarında Ses Kullanımı

Bu eğitim, DSPy kullanarak ses tabanlı uygulamalar için işlem hatlarının nasıl oluşturulacağını adım adım anlatır.

### Bağımlılıkları Kurun

En güncel DSPy sürümünü kullandığınızdan emin olun:

```shell
pip install -U dspy
```

Ses verilerini işlemek için aşağıdaki bağımlılıkları kurun:

```shell
pip install datasets soundfile torch==2.0.1+cu118 torchaudio==2.0.2+cu118
```


### Spoken-SQuAD Veri Kümesini Yükleyin

Bu eğitim gösterimi için, soru-cevap amacıyla kullanılan konuşma sesli pasajları içeren Spoken-SQuAD veri kümesini ([Resmî](https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD) ve [HuggingFace sürümü](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test)) kullanacağız:

```python
import random
import dspy
from dspy.datasets import DataLoader

kwargs = dict(fields=("context", "instruction", "answer"), input_keys=("context", "instruction"))
spoken_squad = DataLoader().from_huggingface(dataset_name="AudioLLMs/spoken_squad_test", split="train", trust_remote_code=True, **kwargs)

random.Random(42).shuffle(spoken_squad)
spoken_squad = spoken_squad[:100]

split_idx = len(spoken_squad) // 2
trainset_raw, testset_raw = spoken_squad[:split_idx], spoken_squad[split_idx:]
```

### Ses Verisini Ön İşleyin

Veri kümesindeki ses klipleri, karşılık gelen örnekleme oranlarıyla birlikte bayt dizilerine dönüştürülmek üzere bir miktar ön işleme gerektirir.

```python
def preprocess(x):
    audio = dspy.Audio.from_array(x.context["array"], x.context["sampling_rate"])
    return dspy.Example(
        passage_audio=audio,
        question=x.instruction,
        answer=x.answer
    ).with_inputs("passage_audio", "question")

trainset = [preprocess(x) for x in trainset_raw]
testset = [preprocess(x) for x in testset_raw]

len(trainset), len(testset)
```

## Konuşmalı soru-cevap için DSPy programı

Soruları doğrudan yanıtlamak için ses girdilerini kullanan basit bir DSPy programı tanımlayalım. Bu, [BasicQA](https://dspy.ai/cheatsheet/?h=basicqa#dspysignature) görevine çok benzer; tek fark, pasaj bağlamının modelin dinleyip soruyu yanıtlaması için bir ses dosyası olarak verilmesidir:

```python
class SpokenQASignature(dspy.Signature):
    """Soruyu ses klibine dayanarak yanıtla."""
    passage_audio: dspy.Audio = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc = '1 ile 5 kelime arasında, kısa olgusal cevap')

spoken_qa = dspy.ChainOfThought(SpokenQASignature)

```

Şimdi ses girdisini işleyebilen LLM’imizi yapılandıralım.

```python
dspy.configure(lm=dspy.LM(model='gpt-4o-mini-audio-preview-2024-12-17'))
```

Not: İmzalarda `dspy.Audio` kullanmak, sesi doğrudan modele geçirmenizi sağlar.

### Değerlendirme Metrğini Tanımlayın

Verilen referans yanıtlarla karşılaştırıldığında cevap doğruluğunu ölçmek için Exact Match metriğini (`dspy.evaluate.answer_exact_match`) kullanacağız:

```python
evaluate_program = dspy.Evaluate(devset=testset, metric=dspy.evaluate.answer_exact_match,display_progress=True, num_threads = 10, display_table=True)

evaluate_program(spoken_qa)
```

### DSPy ile Optimize Edin

Bu ses tabanlı programı, herhangi bir DSPy optimizer ile, diğer DSPy programlarında olduğu gibi optimize edebilirsiniz.

Not: Ses tokenları maliyetli olabilir; bu nedenle `dspy.BootstrapFewShotWithRandomSearch` veya `dspy.MIPROv2` gibi optimizer’ları, optimizer’ın varsayılan parametrelerine göre daha az aday / deneme ve 0-2 few-shot örnek ile temkinli biçimde yapılandırmanız önerilir.

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(metric = dspy.evaluate.answer_exact_match, max_bootstrapped_demos=2, max_labeled_demos=2, num_candidate_programs=5)

optimized_program = optimizer.compile(spoken_qa, trainset = trainset)

evaluate_program(optimized_program)
```

```python
prompt_lm = dspy.LM(model='gpt-4o-mini') #NOT - bu, MIPROv2 talimat adayı önerisini yönlendiren LLM’dir
optimizer = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", prompt_model = prompt_lm)

#NOT - MIPROv2’nin veri kümesi özetleyicisi, veri kümesindeki ses dosyalarını işleyemez; bu nedenle data_aware_proposer özelliğini kapatıyoruz
optimized_program = optimizer.compile(spoken_qa, trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2, data_aware_proposer=False)

evaluate_program(optimized_program)
```

Bu küçük alt kümede, MIPROv2 temel performansa göre yaklaşık %10’luk bir iyileşme sağladı.

---

Artık DSPy içinde ses girdisi destekleyen bir LLM’in nasıl kullanılacağını gördüğümüze göre, kurulumu tersine çevirelim.

Sıradaki görevde, bir metinden sese model için istemler üretmek üzere standart metin tabanlı bir LLM kullanacağız ve ardından üretilen konuşmanın kalitesini sonraki bir görev açısından değerlendireceğiz. Bu yaklaşım, `gpt-4o-mini-audio-preview-2024-12-17` gibi bir LLM’den doğrudan ses üretmesini istemekten genellikle daha maliyet etkindir; aynı zamanda daha yüksek kaliteli konuşma çıktısı için optimize edilebilen bir işlem hattı kurulmasını da sağlar.

### CREMA-D Veri Kümesini Yükleyin

Bu eğitim gösterimi için, seçilmiş katılımcıların aynı cümleyi altı hedef duygudan biriyle söylediği ses kliplerini içeren CREMA-D veri kümesini ([Resmî](https://github.com/CheyneyComputerScience/CREMA-D) ve [HuggingFace sürümü](https://huggingface.co/datasets/myleslinder/crema-d)) kullanacağız: neutral, happy, sad, anger, fear ve disgust.

```python
from collections import defaultdict

label_map = ['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust']

kwargs = dict(fields=("sentence", "label", "audio"), input_keys=("sentence", "label"))
crema_d = DataLoader().from_huggingface(dataset_name="myleslinder/crema-d", split="train", trust_remote_code=True, **kwargs)

def preprocess(x):
    return dspy.Example(
        raw_line=x.sentence,
        target_style=label_map[x.label],
        reference_audio=dspy.Audio.from_array(x.audio["array"], x.audio["sampling_rate"])
    ).with_inputs("raw_line", "target_style")

random.Random(42).shuffle(crema_d)
crema_d = crema_d[:100]

random.seed(42)
label_to_indices = defaultdict(list)
for idx, x in enumerate(crema_d):
    label_to_indices[x.label].append(idx)

per_label = 100 // len(label_map)
train_indices, test_indices = [], []
for indices in label_to_indices.values():
    selected = random.sample(indices, min(per_label, len(indices)))
    split = len(selected) // 2
    train_indices.extend(selected[:split])
    test_indices.extend(selected[split:])

trainset = [preprocess(crema_d[idx]) for idx in train_indices]
testset = [preprocess(crema_d[idx]) for idx in test_indices]
```

## Hedef duygu ile konuşma için TTS talimatları üreten DSPy işlem hattı

Şimdi, TTS modelini hem bir metin satırıyla hem de o satırın nasıl söyleneceğine dair bir talimatla yönlendirerek duygusal olarak ifade gücü yüksek konuşmalar üreten bir işlem hattısı kuracağız.
Bu görevin amacı, veri kümesindeki referans sesin duygu ve stiline uyan TTS çıktısı üretimini yönlendirecek istemleri DSPy ile üretmektir.

Önce, belirtilen bir duygu veya stille konuşulan ses üretmek üzere TTS üreticisini kuralım.
`gpt-4o-mini-tts` modelini kullanıyoruz; çünkü bu model, ham girdi ve konuşma biçimi ile istem vermeyi destekler ve `dspy.Audio` ile işlenen `.wav` dosyası biçiminde ses yanıtı üretir.
Ayrıca TTS çıktıları için bir önbellek de kuruyoruz.

```python
import os
import base64
import hashlib
from openai import OpenAI

CACHE_DIR = ".audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def hash_key(raw_line: str, prompt: str) -> str:
    return hashlib.sha256(f"{raw_line}|||{prompt}".encode("utf-8")).hexdigest()

def generate_dspy_audio(raw_line: str, prompt: str) -> dspy.Audio:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    key = hash_key(raw_line, prompt)
    wav_path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(wav_path):
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral", #NOT - bu, OpenAI’nin sunduğu 11 TTS sesinden herhangi biri olarak yapılandırılabilir - https://platform.openai.com/docs/guides/text-to-speech#voice-options.
            input=raw_line,
            instructions=prompt,
            response_format="wav"
        )
        with open(wav_path, "wb") as f:
            f.write(response.content)
    with open(wav_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return dspy.Audio(data=encoded, format="wav")
```

Şimdi TTS talimatları üretmek için DSPy programını tanımlayalım. Bu program için yalnızca talimat ürettiğimizden, yeniden standart metin tabanlı LLM’leri kullanabiliriz.

```python
class EmotionStylePromptSignature(dspy.Signature):
    """TTS modelinin verilen satırı hedef duygu veya stille söylemesini sağlayacak bir OpenAI TTS talimatı üret."""
    raw_line: str = dspy.InputField()
    target_style: str = dspy.InputField()
    openai_instruction: str = dspy.OutputField()

class EmotionStylePrompter(dspy.Module):
    def __init__(self):
        self.prompter = dspy.ChainOfThought(EmotionStylePromptSignature)

    def forward(self, raw_line, target_style):
        out = self.prompter(raw_line=raw_line, target_style=target_style)
        audio = generate_dspy_audio(raw_line, out.openai_instruction)
        return dspy.Prediction(audio=audio)
    
dspy.configure(lm=dspy.LM(model='gpt-4o-mini'))
```

### Değerlendirme Metrğini Tanımlayın

Referans ses karşılaştırmaları, özellikle duygusal ifade söz konusu olduğunda, konuşmanın öznel değerlendirme farklılıkları nedeniyle genel olarak kolay olmayan bir görevdir. Bu eğitimde, nesnel değerlendirme amacıyla embedding tabanlı bir benzerlik metriği kullanıyoruz; bunun için sesi embedding’lere dönüştürmek üzere Wav2Vec 2.0’dan yararlanıyor ve referans ses ile üretilen ses arasında kosinüs benzerliği hesaplıyoruz. Ses kalitesini daha doğru değerlendirmek için insan geri bildirimi veya algısal metrikler daha uygun olacaktır.

```python
import torch
import torchaudio
import soundfile as sf
import io

bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().eval()

def decode_dspy_audio(dspy_audio):
    audio_bytes = base64.b64decode(dspy_audio.data)
    array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return torch.tensor(array).unsqueeze(0)

def extract_embedding(audio_tensor):
    with torch.inference_mode():
        return model(audio_tensor)[0].mean(dim=1)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def audio_similarity_metric(example, pred, trace=None):
    ref_audio = decode_dspy_audio(example.reference_audio)
    gen_audio = decode_dspy_audio(pred.audio)

    ref_embed = extract_embedding(ref_audio)
    gen_embed = extract_embedding(gen_audio)

    score = cosine_similarity(ref_embed, gen_embed)

    if trace is not None:
        return score > 0.8 
    return score

evaluate_program = dspy.Evaluate(devset=testset, metric=audio_similarity_metric, display_progress=True, num_threads = 10, display_table=True)

evaluate_program(EmotionStylePrompter())
```

DSPy programının hangi talimatları ürettiğini ve buna karşılık gelen skoru görmek için bir örneğe bakabiliriz:

```python
program = EmotionStylePrompter()

pred = program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)

print(audio_similarity_metric(testset[1], pred)) #0.5725605487823486

dspy.inspect_history(n=1)
```

TTS Talimatı:
```text
Aşağıdaki satırı tiksinti tonuyla söyle: Saat on bir.
```


```python
from IPython.display import Audio

audio_bytes = base64.b64decode(pred.audio.data)
array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
Audio(array, rate=rate)
```

Talimat hedef duyguyu belirtiyor, ancak bunun ötesinde çok bilgilendirici değil. Ayrıca bu örnek için ses skorunun da çok yüksek olmadığını görebiliyoruz. Bakalım bu işlem hattını optimize ederek daha iyisini yapabilecek miyiz.

### DSPy ile Optimize Edin

Aşağı akış görev hedefini iyileştirip daha yüksek kaliteli TTS talimatları üretmek ve böylece daha doğru ve daha ifade gücü yüksek ses üretimleri elde etmek için `dspy.MIPROv2` kullanabiliriz:

```python
prompt_lm = dspy.LM(model='gpt-4o-mini')

teleprompter = dspy.MIPROv2(metric=audio_similarity_metric, auto="light", prompt_model = prompt_lm)

optimized_program = teleprompter.compile(EmotionStylePrompter(),trainset=trainset)

evaluate_program(optimized_program)

```

Optimize edilmiş programın nasıl performans gösterdiğine bakalım:

```python
pred = optimized_program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)

print(audio_similarity_metric(testset[1], pred)) #0.6691027879714966

dspy.inspect_history(n=1)
```

MIPROv2 ile Optimize Edilmiş Program Talimatı:
```text 
TTS modelinin verilen satırı hedef duygu veya stille söylemesini sağlayacak bir OpenAI TTS talimatı üret; bunu, konuşmacı görevle ilgili bir [uygun persona ekleyin, ör. "öfkeli müşteri", "sinirli patron" vb.] imiş gibi yap. Talimat, hedef duyguyu aktarmak için konuşmacının ses tonunu, perdesini ve diğer özelliklerini belirtmelidir.
```

TTS Talimatı:
```text
Girdi metni olan "Saat on bir." için aşağıdaki özelliklere sahip bir metinden sese sentezi üret:
- Ton: Tiksinmiş
- Perde: Yüksek perdeli, hafif burunsu
- Vurgu: Tiksinti ve kaçınma duygusunu aktarmak için kelimeleri özellikle vurgula
- Ses yüksekliği: Orta ile yüksek arası; konuşmacının güçlü olumsuz duygularını aktarmak için sonda yükselen bir tonlama hissi olsun
- Konuşmacı: Az önce bozulmuş bir yemek servis edilmiş bir karakter gibi, açıkça görülebilir ve duyulabilir biçimde tiksinmiş bir kişi.
```

```python
from IPython.display import Audio

audio_bytes = base64.b64decode(pred.audio.data)
array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
Audio(array, rate=rate)
```

MIPROv2’nin talimat ayarlaması, genel görev hedefine daha fazla nüans kattı; TTS talimatının nasıl tanımlanması gerektiğine dair daha fazla ölçüt ekledi ve bunun sonucunda üretilen talimat, konuşma prozodisindeki çeşitli etkenler açısından çok daha özgül hale gelerek daha yüksek bir benzerlik skoru üretti.
