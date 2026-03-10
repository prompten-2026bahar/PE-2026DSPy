# Model Bağlam Protokolü (Model Context Protocol - MCP)

[Model Bağlam Protokolü (MCP)](https://modelcontextprotocol.io/), uygulamaların dil modellerine nasıl bağlam sağlayacağını standartlaştıran açık bir protokoldür. DSPy, MCP'yi destekleyerek herhangi bir MCP sunucusundaki araçları DSPy ajanlarıyla kullanmanıza olanak tanır.

## Kurulum

MCP destekli DSPy kurulumunu gerçekleştirin:

```bash
pip install -U "dspy[mcp]"
```

## Genel Bakış

MCP şunları yapmanıza olanak tanır:

- **Standartlaştırılmış araçları kullanın** - Herhangi bir MCP uyumlu sunucuya bağlanın.
- **Araçları farklı yığınlar (stacks) arasında paylaşın** - Aynı araçları farklı çerçeveler (frameworks) genelinde kullanın.
- **Entegrasyonu basitleştirin** - MCP araçlarını tek bir satırla DSPy araçlarına dönüştürün.

DSPy, MCP sunucu bağlantılarını doğrudan yönetmez. Bağlantıyı kurmak için `mcp` kütüphanesinin istemci arayüzlerini kullanabilir ve MCP araçlarını DSPy araçlarına dönüştürmek için `mcp.ClientSession` nesnesini `dspy.Tool.from_mcp_tool` metoduna iletebilirsiniz.

## DSPy ile MCP Kullanımı

### 1. HTTP Sunucusu (Uzak)

HTTP üzerinden uzak MCP sunucuları için akış yapılabilir (streamable) HTTP aktarımını kullanın:

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # HTTP MCP sunucusuna bağlanın
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # Oturumu başlatın
            await session.initialize()

            # Araçları listeleyin ve dönüştürün
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # ReAct ajanını oluşturun ve kullanın
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="Tokyo'da hava durumunu kontrol et")
            print(result.result)

asyncio.run(main())
```

### 2. Stdio Sunucusu (Yerel İşlem)

MCP'yi kullanmanın en yaygın yolu, stdio aracılığıyla iletişim kuran yerel bir sunucu işlemi kullanmaktır:

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Stdio sunucusunu yapılandırın
    server_params = StdioServerParameters(
        command="python",                    # Çalıştırılacak komut
        args=["path/to/your/mcp_server.py"], # Sunucu betiği yolu
        env=None,                            # İsteğe bağlı ortam değişkenleri
    )

    # Sunucuya bağlanın
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Oturumu başlatın
            await session.initialize()

            # Kullanılabilir araçları listeleyin
            response = await session.list_tools()

            # MCP araçlarını DSPy araçlarına dönüştürün
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # Araçlarla bir ReAct ajanı oluşturun
            class QuestionAnswer(dspy.Signature):
                """Mevcut araçları kullanarak soruları yanıtlayın."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # Ajanı kullanın
            result = await react_agent.acall(
                question="25 + 17 kaç eder?"
            )
            print(result.answer)

# Asenkron fonksiyonu çalıştırın
asyncio.run(main())
```

## Araç Dönüştürme

DSPy, MCP araçlarından DSPy araçlarına dönüşümü otomatik olarak yönetir:

```python
# Oturumdan gelen MCP aracı
mcp_tool = response.tools[0]

# DSPy aracına dönüştürün
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# DSPy aracı şunları korur:
# - Araç adı ve açıklaması
# - Parametre şemaları ve türleri
# - Argüman açıklamaları
# - Asenkron yürütme desteği

# Herhangi bir DSPy aracı gibi kullanın
result = await dspy_tool.acall(param1="değer", param2=123)
```

## Daha Fazla Bilgi Edinin



- [MCP Resmi Dokümantasyonu](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP Eğitimi](https://dspy.ai/tutorials/mcp/)
- [DSPy Araçlar Dokümantasyonu](./tools.md)

DSPy'deki MCP entegrasyonu, herhangi bir MCP sunucusundaki standartlaştırılmış araçları kullanmayı kolaylaştırarak minimum kurulumla güçlü ajan yetenekleri sağlar.