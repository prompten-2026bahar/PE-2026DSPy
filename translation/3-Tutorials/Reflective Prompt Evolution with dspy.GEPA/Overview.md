# GEPA ile Yansıtıcı İstem Evrimi

Bu bölüm, DSPy için yansıtıcı bir istem optimize edicisi olan GEPA’yı tanıtır. GEPA, DSPy programının izleği üzerinde düşünmek için dil modelinin yeteneğinden yararlanarak çalışır; nelerin iyi gittiğini, nelerin iyi gitmediğini ve nelerin iyileştirilebileceğini belirler. Bu yansıma temelinde GEPA yeni istemler önerir, evrilmiş istem adaylarından oluşan bir ağaç kurar ve optimizasyon ilerledikçe iyileştirmeleri biriktirir. GEPA, yalnızca skaler metriğe değil, alana özgü metinsel geri bildirime de dayanabildiği için çoğu zaman çok az sayıda rollout ile yüksek performanslı istemler önerebilir. GEPA, [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) makalesinde tanıtılmıştır ve dahili olarak [gepa-ai/gepa](https://github.com/gepa-ai/gepa) tarafından sağlanan GEPA uygulamasını kullanan `dspy.GEPA` olarak उपलब्धdır.

## `dspy.GEPA` Eğitimleri

### [AIME (Matematik) için GEPA](../gepa_aime/index.ipynb)
Bu eğitim, GEPA’nın tek bir `dspy.ChainOfThought` tabanlı programı optimize ederek GPT-4.1 Mini ile AIME 2025 üzerinde %10’luk kazanç elde etmesini nasıl sağlayabildiğini inceler.

### [Kurumsal Görevler için Yapılandırılmış Bilgi Çıkarımında GEPA](../gepa_facilitysupportanalyzer/index.ipynb)
Bu eğitim, GEPA’nın kurumsal bir ortamda yapılandırılmış bilgi çıkarımı ve sınıflandırmaya yönelik üç parçalı bir görevde GPT-4.1 Nano’nun performansını artırmak için kestirici düzeyinde geri bildirimden nasıl yararlandığını inceler.

### [Gizlilik Bilinçli Yetki Devri için GEPA](../gepa_papillon/index.ipynb)
Bu eğitim, GEPA’nın bir yargıç olarak LLM metriğinin sağladığı basit geri bildirimi kullanırken, yalnızca 1 yineleme gibi kısa bir sürede nasıl hızla iyileşebildiğini inceler. Eğitim ayrıca, toplu metriklerin alt bileşenlere ayrılmış dökümünü gösteren metinsel geri bildirimin GEPA’ya nasıl fayda sağladığını ve böylece yansıtıcı dil modelinin görevin hangi yönlerinin iyileştirilmesi gerektiğini belirleyebildiğini de ele alır.

### [Kod Arka Kapı Sınıflandırması (AI control) için GEPA](../gepa_trusted_monitor/index.ipynb)
Bu eğitim, GEPA’nın `dspy.GEPA` ve karşılaştırmalı bir metrik kullanarak, daha büyük bir dil modeli tarafından yazılmış koddaki arka kapıları belirlemek için bir GPT-4.1 Nano sınıflandırıcısını nasıl optimize edebildiğini inceler. Karşılaştırmalı metrik, istem optimize edicisinin kod içinde arka kapıya işaret eden sinyalleri belirleyen bir istem oluşturmasına olanak tanır ve pozitif örnekleri negatif örneklerden ayırır.
