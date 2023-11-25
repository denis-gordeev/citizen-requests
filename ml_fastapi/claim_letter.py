from gigachat import GigaChat
def build_prompt_letter(incident, subject, person):
    prompt = f" Модератор получил следующий текст:  '{incident}'"
    prompt += f" Модератор определил тему письма:  '{subject}.'"
    prompt += f" Модератор определил организацию исполнителя:  '{person}.'"
    prompt += f" Напиши запрос в адрес организации  от имени модератора о проблеме из текста. В запрос включи: кратко о проблеме, требование плана устранения проблемы. проблема из текста должна быть описана от третьего лица."
    return prompt

def giga_letter(incident, subject, person):
    credentials='YzdmMDg1YmMtYmU2Mi00ZDdiLTkxZTItODRhMmU5NGE0YTI4OjllZTIwZTRhLWMxOWUtNDMyZS1hYTg2LTk3Y2Y4MjVkOTAyYQ=='
    with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
        giga.temperature=0
        response = giga.chat(build_prompt_letter(incident, subject, person))
    text = response.choices[0].message.content
    return text
