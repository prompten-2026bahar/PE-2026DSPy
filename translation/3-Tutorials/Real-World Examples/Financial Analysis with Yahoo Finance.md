# DSPy ReAct ve Yahoo Finance News ile Finansal Analiz

Bu eğitim, gerçek zamanlı piyasa analizi için [LangChain’in Yahoo Finance News aracını](https://python.langchain.com/docs/integrations/tools/yahoo_finance_news/) DSPy ReAct ile kullanarak bir finansal analiz ajanının nasıl oluşturulacağını gösterir.

## Ne İnşa Edeceksiniz

Haberleri getiren, duygu analizi yapan ve yatırım içgörüleri sunan bir finans ajanı.

## Kurulum

```bash
pip install dspy langchain langchain-community yfinance
```

## Adım 1: LangChain Aracını DSPy’ye Dönüştürme

```python
import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf

# DSPy'yi yapılandır
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm, allow_tool_async_sync_conversion=True)

# LangChain Yahoo Finance aracını DSPy'ye dönüştür
yahoo_finance_tool = YahooFinanceNewsTool()
finance_news_tool = Tool.from_langchain(yahoo_finance_tool)
```

## Adım 2: Destekleyici Finansal Araçlar Oluşturma

```python
def get_stock_price(ticker: str) -> str:
    """Güncel hisse fiyatını ve temel bilgileri getir."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")

        if hist.empty:
            return f"{ticker} için veri alınamadı"

        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

        result = {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_percent": round(change_pct, 2),
            "company": info.get('longName', ticker)
        }

        return json.dumps(result)
    except Exception as e:
        return f"Hata: {str(e)}"

def compare_stocks(tickers: str) -> str:
    """Birden fazla hisseyi karşılaştırır (virgülle ayrılmış)."""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        comparison = []

        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

                comparison.append({
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change_percent": round(change_pct, 2)
                })

        return json.dumps(comparison)
    except Exception as e:
        return f"Hata: {str(e)}"
```

## Adım 3: Finansal ReAct Ajanını Oluşturma

```python
class FinancialAnalysisAgent(dspy.Module):
    """Yahoo Finance verilerini kullanan finansal analiz için ReAct ajanı."""

    def __init__(self):
        super().__init__()

        # Tüm araçları birleştir
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
            get_stock_price,
            compare_stocks
        ]

        # ReAct'i başlat
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response",
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        return self.react(financial_query=financial_query)
```

## Adım 4: Finansal Analizi Çalıştırma

```python
def run_financial_demo():
    """Finansal analiz ajanının demosu."""

    # Ajanı başlat
    agent = FinancialAnalysisAgent()

    # Örnek sorgular
    queries = [
        "Apple (AAPL) hakkındaki en son haberler neler ve bunlar hisse fiyatını nasıl etkileyebilir?",
        "AAPL, GOOGL ve MSFT performansını karşılaştır",
        "Tesla ile ilgili son haberleri bul ve duygu analizini yap"
    ]

    for query in queries:
        print(f"Sorgu: {query}")
        response = agent(financial_query=query)
        print(f"Analiz: {response.analysis_response}")
        print("-" * 50)

# Demoyu çalıştır
if __name__ == "__main__":
    run_financial_demo()
```

## Örnek Çıktı

Ajanı “Apple hakkındaki en son haberler neler?” gibi bir sorguyla çalıştırdığınızda şunları yapacaktır:

1. Son Apple haberlerini getirmek için Yahoo Finance News aracını kullanır
2. Güncel hisse fiyatı verilerini alır
3. Bilgileri analiz eder ve içgörüler sunar

**Örnek Yanıt:**
```
Analiz: Apple’ın (AAPL) mevcut fiyatı 196,58 $ ve %0,48’lik hafif artış göz önüne alındığında, hissenin piyasada istikrarlı performans gösterdiği görülüyor. Ancak en son haberlere erişilememesi, yatırımcı duyarlılığını ve hisse fiyatını etkileyebilecek önemli gelişmelerin bilinmediği anlamına geliyor. Yatırımcılar, özellikle Microsoft (MSFT) gibi olumlu eğilim gösteren diğer teknoloji hisseleriyle karşılaştırıldığında, Apple’ın performansını etkileyebilecek yaklaşan duyuruları veya piyasa trendlerini yakından takip etmelidir.
```

## Async Araçlarla Çalışma

Birçok LangChain aracı, daha iyi performans için async işlemler kullanır. Async araçlar hakkında ayrıntılar için [Araçlar dokümantasyonuna](../../learn/programming/tools.md#async-tools) bakın.

## Temel Avantajlar

- **Araç Entegrasyonu**: LangChain araçlarını DSPy ReAct ile sorunsuz biçimde birleştirme
- **Gerçek Zamanlı Veri**: Güncel piyasa verilerine ve haberlere erişim
- **Genişletilebilirlik**: Daha fazla finansal analiz aracı eklemenin kolay olması
- **Akıllı Akıl Yürütme**: ReAct çerçevesi adım adım analiz sağlar

Bu eğitim, DSPy’nin ReAct çerçevesinin LangChain’in finansal araçlarıyla birlikte çalışarak akıllı piyasa analiz ajanları oluşturduğunu gösterir.
