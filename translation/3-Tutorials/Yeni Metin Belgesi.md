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
