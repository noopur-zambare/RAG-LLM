FROM python:3
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN ollama run deepseek-r1:1.5b
    ollama run llama3.3
    ollama run mixtral

COPY . .

CMD [ "streamlit", "run", "app.py" ]
